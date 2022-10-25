from . import RewardLearner
import torch
from matplotlib.pyplot import subplots


class RewardTree(RewardLearner):
    """
    Preference-based reward learner with a tree model.
    """
    def __init__(self, features_and_thresholds:dict, max_num_eps:int, max_ep_length:int=None, embed_by_ep:bool=True):
        # Here self.model learns individual predictions for transitions in the dataset
        # These are used as inputs to the tree growth process
        RewardLearner.__init__(self,
            model=EmbeddingModel(
                num_embeddings=max_num_eps*(1 if embed_by_ep else max_ep_length),
                lr=1e-1),
            embed_by_ep=embed_by_ep)
        self.features_and_thresholds = {f: t.to(self.device) for f, t in features_and_thresholds.items()}
    
    def __call__(self, states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor) -> torch.Tensor:
        return self.transitions_to_visits(states, actions, next_states).float() @ self.r_mean

    def transitions_to_visits(self, states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor, one_hot:bool=True) -> torch.Tensor:
        """
        Given tensors of transitions, recursively propagate through the tree to get leaf visits.
        """
        assert states.dim() == next_states.dim() == 2, "Currently only works with 2D"
        shape, num_leaves = states.shape, len(self.leaves)
        if one_hot:
            visits = torch.full((*shape[:-1], num_leaves), -1, device=self.device)
            oh = torch.eye(num_leaves, dtype=int, device=self.device)
        else: visits = torch.full(shape[:-1], -1, device=self.device)
        def propagate(node, ind):
            if len(ind) == 0: return
            if node in self.leaves: # At a leaf, store the leaf num for all remaining indices
                x = self.leaves.index(node)
                visits[ind] = oh[x] if one_hot else x
            else: # At an internal node, split the indices based on the split threshold
                left_mask = node(states[ind], actions[ind], next_states[ind])
                propagate(node.left, ind[left_mask]); propagate(node.right, ind[~left_mask])
        propagate(self.root, torch.arange(shape[0], device=self.device))
        return visits

    def train(self, max_num_leaves:int=2, loss_func:str="0-1", num_batches:int=500, batch_size:int=32):
        """
        Complete model induction process from paper "Reward Learning with Trees: Methods and Evaluation".
        """
        assert loss_func in {"0-1", "bce"}
        # Estimate rewards for transitions in the dataset by gradient descent on BCE loss
        for i in range(num_batches): print(i, self.update_on_batch(batch_size).item())
        self.normalise()
        with torch.no_grad(): # No autograd needed during tree induction
            r = (self.indices_to_rewards() - self.shift) / self.scale
            # Extract convenient representations of the preference dataset for use in training
            i_j    = [(self.get_indices(p.i), self.get_indices(p.j)) for p in self.preferences]
            ind    = torch.unique(torch.cat([torch.cat([i, j]) for i, j in i_j]))
            y      = torch.tensor([p.y for p in self.preferences], device=self.device)
            y_sign = (y - 0.5).sign()
            w      = torch.tensor([p.w for p in self.preferences], device=self.device)
            # Initialise a blank tree
            self.root = Node(ind, self.features_and_thresholds)
            self.leaves = [self.root]
            self.visits = torch.ones((1, self.states.shape[0]), dtype=bool, device=self.device)
            self.r_mean = torch.zeros(1, device=self.device)
            current_loss_0_1 = 1.
            current_loss_bce = (self.bce_loss_noreduce(0.5 * torch.ones_like(y), y) * w).mean()
            # Growth stage
            while len(self.leaves) < max_num_leaves:
                # _, ax = subplots(1, len(self.leaves), squeeze=False); ax = ax.flatten()
                candidates = []
                # Iterate through each leaf in the current tree
                for l, leaf in enumerate(self.leaves):
                    # Gather visits and mean rewards from all other leaves
                    visits_other = torch.cat([self.visits[:l], torch.zeros((2, self.states.shape[0]), dtype=bool, device=self.device),
                                              self.visits[l+1:]]).unsqueeze(0)
                    r_mean_other = torch.cat([self.r_mean[:l], torch.zeros(2, device=self.device), self.r_mean[l+1:]]).unsqueeze(0)
                    # Iterate through each splitting feature
                    for f in leaf.features_and_thresholds:
                        num_thresholds = len(leaf.features_and_thresholds[f])
                        if f not in leaf.left_mask: # Only do this if haven't previously computed and cached
                            # leaf.left_mask[f] is a (num_thresholds x len(leaf.ind)) binary matrix,
                            # indicating whether the feature value for each transition is less than each threshold
                            feature_values = f(self.states[leaf.ind], self.actions[leaf.ind], self.next_states[leaf.ind])
                            leaf.left_mask[f] = (feature_values.unsqueeze(0) < leaf.features_and_thresholds[f].unsqueeze(1)).squeeze()
                            # Use left mask to compute visits and mean rewards for left and right children, for each threshold
                            leaf.visits[f] = torch.zeros((num_thresholds, 2, len(self.states)), device=self.device)
                            leaf.r_mean[f] = torch.zeros((num_thresholds, 2), device=self.device)
                            for t, left_mask in enumerate(leaf.left_mask[f]):
                                ind_left, ind_right = leaf.ind[left_mask], leaf.ind[~left_mask]
                                leaf.visits[f][t, 0, ind_left ] = 1
                                leaf.visits[f][t, 1, ind_right] = 1
                                if len(ind_left):  leaf.r_mean[f][t, 0] = r[ind_left ].mean()
                                if len(ind_right): leaf.r_mean[f][t, 1] = r[ind_right].mean()
                        # Combine visits and mean rewards for children with those for other leaves
                        visits_all = visits_other.tile((num_thresholds, 1, 1))
                        r_mean_all = r_mean_other.tile((num_thresholds, 1))
                        visits_all[:, l:l+2, :] = leaf.visits[f]
                        r_mean_all[:, l:l+2   ] = leaf.r_mean[f]
                        # Compute mean reward differences for all pairs in the preference dataset
                        diff = torch.zeros((num_thresholds, len(self.preferences)), device=self.device)
                        for k, (i, j) in enumerate(i_j):
                            diff[:, k] = ((visits_all[..., i].sum(dim=-1) * r_mean_all).sum(dim=1) / len(i)) - \
                                         ((visits_all[..., j].sum(dim=-1) * r_mean_all).sum(dim=1) / len(j))
                        # Compute losses (0-1 and BCE) and identify the best split using one of these
                        loss_0_1 = ((diff.sign() != y_sign) * w).mean(dim=1)
                        loss_bce = (self.bce_loss_noreduce(self.sigmoid(diff), y.unsqueeze(0).tile((num_thresholds, 1))) * w).mean(dim=1)
                        if loss_func == "0-1": loss_reduction = current_loss_0_1 - loss_0_1
                        else:                  loss_reduction = current_loss_bce - loss_bce
                        best_split = loss_reduction.argmax()
                        if loss_reduction[best_split] > 0: # Only keep split if loss is reduced
                            candidates.append((loss_0_1[best_split], loss_bce[best_split], l, f, best_split))
                        # ax[l].plot(leaf.features_and_thresholds[f], loss_reduction)
                if len(candidates) == 0: break # If loss reduction not possible, stop growth
                # Identify and make the best split across all leaves and features
                current_loss_0_1, current_loss_bce, l, f, best_split = sorted(candidates, key=lambda c: c[0 if loss_func == "0-1" else 1])[0]
                node = self.leaves[l]
                node.split(f, best_split)
                print(f"{len(self.leaves)}: Split leaf {l} at {f.__name__}={node.threshold} (loss_0_1={current_loss_0_1}, loss_bce={current_loss_bce})")
                # Update attributes for use in future growth
                self.leaves = self.leaves[:l] + [node.left, node.right] + self.leaves[l+1:]
                self.visits = torch.cat([self.visits[:l], node.visits[f][best_split], self.visits[l+1:]])
                self.r_mean = torch.cat([self.r_mean[:l], node.r_mean[f][best_split], self.r_mean[l+1:]])
            #####################
            # TODO: pruning stage
            #####################

    def to_hyperrectangles(self):
        """
        Convert tree into a form that enables visual and textual representation.
        Requires https://github.com/tombewley/hyperrectangles.
        """
        from hyperrectangles import Space, Node as hr_Node, Tree
        feature_order = [f for f in self.features_and_thresholds]
        def copy_tree(node, hr_node):
            if node in self.leaves: hr_node.mean[-1] = self.r_mean[self.leaves.index(node)]
            else:
                hr_node._do_split(feature_order.index(node.feature), node.threshold.item())
                copy_tree(node.left,  hr_node.left )
                copy_tree(node.right, hr_node.right)
            return hr_node
        return Tree("reward_tree", copy_tree(self.root, hr_Node(Space([f.__name__ for f in feature_order] + ["reward"]))), None, None)


class EmbeddingModel(torch.nn.Embedding):
    def __init__(self, *args, optimiser=torch.optim.Adam, lr=1e-1, **kwargs):
        super(EmbeddingModel, self).__init__(embedding_dim=1, *args, **kwargs)
        self.optimiser = optimiser(self.parameters(), lr=lr)

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        return super().forward(i).squeeze()

    def optimise(self, loss): 
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


class Node:
    """
    Class for a node in the tree.
    """
    def __init__(self, ind, features_and_thresholds):
        # On creation
        self.ind, self.features_and_thresholds = ind, features_and_thresholds
        # Pseudo
        self.left_mask, self.visits, self.r_mean = {}, {}, {}
        # On self.split()
        self.feature, self.threshold = None, None
        self.left, self.right = None, None

    def __call__(self, states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor) -> torch.Tensor:
        return self.feature(states, actions, next_states) < self.threshold

    def split(self, f, split):
        self.feature = f
        # Create two child nodes
        left_mask = self.left_mask[f][split]
        self.left  = Node(self.ind[left_mask ], self.features_and_thresholds.copy())
        self.right = Node(self.ind[~left_mask], self.features_and_thresholds.copy())
        # Simultaneously set split threshold for this leaf, and candidate thresholds for children
        self.left.features_and_thresholds[f], self.threshold, self.right.features_and_thresholds[f] = \
            torch.split(self.features_and_thresholds[f], [split, 1, len(self.features_and_thresholds[f])-split-1])
        self.threshold = self.threshold.squeeze()
