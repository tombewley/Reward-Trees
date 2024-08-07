from . import RewardLearner
import torch
from numpy import unravel_index
from matplotlib.pyplot import subplots


class RewardTree(RewardLearner):
    """Preference-based reward learner with a tree model."""
    def __init__(self, features_and_thresholds:dict, max_num_eps:int, max_ep_length:int=None, embed_by_ep:bool=True, seed:int=None, **kwargs):
        if seed is not None: torch.manual_seed(seed) # Need to do this here for seeded model initialisation
        # Here self.model learns individual predictions for transitions in the dataset
        # These are used as inputs to the tree growth process
        RewardLearner.__init__(self,
            model=EmbeddingModel(
                num_embeddings=max_num_eps*(1 if embed_by_ep else max_ep_length),
                lr=3e-4),
            embed_by_ep=embed_by_ep, seed=seed, **kwargs)
        self.features_and_thresholds = {f: t.to(self.device) for f, t in features_and_thresholds.items()}
        self.reset()

    def __call__(self, states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor) -> torch.Tensor:
        return self.transitions_to_visits(states, actions, next_states).float() @ self.r_mean

    def transitions_to_visits(self, states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor, one_hot:bool=True) -> torch.Tensor:
        """Given tensors of transitions, recursively propagate through the tree to get leaf visits."""
        shape, num_leaves = states.shape, len(self.leaves)
        flatten_to = states.dim() - self.states.dim()
        if one_hot:
            visits = torch.full((*shape[:flatten_to+1], num_leaves), -1, device=self.device)
            oh = torch.eye(num_leaves, dtype=int, device=self.device)
        else: visits = torch.full(shape[:flatten_to+1], -1, device=self.device)
        s_flat  = states     .flatten(0, flatten_to)
        a_flat  = actions    .flatten(0, flatten_to)
        ns_flat = next_states.flatten(0, flatten_to)
        def propagate(node, ind):
            if len(ind) == 0: return
            if node in self.leaves: # At a leaf, store the leaf num for all remaining indices
                ind_unflat = [torch.tensor(i, device=self.device) for i in unravel_index(ind.cpu().numpy(), shape[:flatten_to+1])]
                x = self.leaves.index(node)
                visits[ind_unflat] = oh[x] if one_hot else x
            else: # At an internal node, split the indices based on the split threshold
                left_mask = node(s_flat[ind], a_flat[ind], ns_flat[ind])
                propagate(node.left, ind[left_mask]); propagate(node.right, ind[~left_mask])
        propagate(self.root, torch.arange(s_flat.shape[0], device=self.device))
        return visits

    def reset(self, ind=None):
        """Initialise a blank tree and optionally populate with indices."""
        self.root = Node(ind, self.features_and_thresholds)
        self.leaves = [self.root]
        self.r_mean = torch.zeros(1, device=self.device)

    def train(self, max_num_leaves:int=2, loss_func:str="0-1", num_batches:int=500, batch_size:int=32, callbacks=None, plot:bool=False):
        """Complete model induction process from paper "Reward Learning with Trees: Methods and Evaluation"."""
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
            self.reset(ind)
            idx_to_leaf = self.transitions_to_visits(self.states, self.actions, self.next_states, one_hot=False)
            current_loss_0_1 = 1.
            current_loss_bce = (self.bce_loss_noreduce(0.5 * torch.ones_like(y), y) * w).mean()
            # Growth stage
            while len(self.leaves) < max_num_leaves:
                if plot: _, ax = subplots(1, len(self.leaves), squeeze=False); ax = ax.flatten()
                candidates = []
                # Iterate through each leaf in the current tree
                for l, leaf in enumerate(self.leaves):
                    # Predefine incremented idx_to_leaf record and expanded mean reward vector
                    idx_to_leaf_other = idx_to_leaf.clone(); idx_to_leaf_other[idx_to_leaf > l] += 1
                    r_mean_other = torch.cat([self.r_mean[:l], torch.empty(2, device=self.device), self.r_mean[l+1:]]).unsqueeze(0)
                    oh = torch.eye(len(self.leaves) + 1, dtype=int, device=self.device)
                    # Iterate through each splitting feature
                    for f in leaf.features_and_thresholds:
                        num_thresholds = len(leaf.features_and_thresholds[f])
                        if f not in leaf.ind_right: # Only do this if haven't previously computed and cached
                            # right_mask is a (num_thresholds x len(leaf.ind)) binary matrix,
                            # indicating whether the feature value for each transition is >= each threshold
                            feature_values = f(self.states[leaf.ind], self.actions[leaf.ind], self.next_states[leaf.ind])
                            right_mask = (feature_values.unsqueeze(0) >= leaf.features_and_thresholds[f].unsqueeze(1))
                            # Use right mask to compute mean rewards for left and right children, for each threshold
                            leaf.r_mean[f] = torch.zeros((num_thresholds, 2), device=self.device)
                            leaf.ind_right[f] = [None for _ in range(num_thresholds)]
                            for t, rm in enumerate(right_mask):
                                ind_left, ind_right = leaf.ind[~rm], leaf.ind[rm]
                                leaf.ind_right[f][t] = ind_right
                                if len(ind_left):  leaf.r_mean[f][t, 0] = r[ind_left ].mean()
                                if len(ind_right): leaf.r_mean[f][t, 1] = r[ind_right].mean()
                        # Combine idx -> leaf mapping and mean rewards for children with those for other leaves
                        idx_to_leaf_all = idx_to_leaf_other.tile((num_thresholds, 1))
                        r_mean_all = r_mean_other.tile((num_thresholds, 1))
                        for t, ind_right in enumerate(leaf.ind_right[f]):
                            idx_to_leaf_all[t, ind_right] += 1
                        r_mean_all[..., l:l+2] = leaf.r_mean[f]
                        # Compute mean reward differences for all pairs in the preference dataset
                        r_pred = torch.gather(r_mean_all, 1, idx_to_leaf_all)
                        diff = torch.stack([self.integrate_utility(r_pred[:, i]) - self.integrate_utility(r_pred[:, j]) for i, j in i_j], dim=1)
                        # # NOTE: this method for computing `diff` is slower but has better numerical precision
                        # visits_all = oh[idx_to_leaf_all]
                        # diff = torch.zeros((num_thresholds, len(self.preferences)), device=self.device)
                        # for k, (i, j) in enumerate(i_j):
                        #     diff[:, k] = ((visits_all[:, i].sum(dim=1) * r_mean_all).sum(dim=1) / len(i)) - \
                        #                  ((visits_all[:, j].sum(dim=1) * r_mean_all).sum(dim=1) / len(j))
                        # Deal with numerical imprecision, which can add significant noise to `loss_0_1`
                        diff[diff.abs() < 1e-5] = 0.
                        # Compute losses (0-1 and BCE) and identify the best split using one of these
                        loss_0_1 = ((diff.sign() != y_sign) * w).mean(dim=1)
                        loss_bce = (self.bce_loss_noreduce(self.sigmoid(diff), y.unsqueeze(0).tile((num_thresholds, 1))) * w).mean(dim=1)
                        if loss_func == "0-1": loss_reduction = current_loss_0_1 - loss_0_1
                        else:                  loss_reduction = current_loss_bce - loss_bce
                        if len(loss_reduction) > 0:
                            best_split = loss_reduction.argmax()
                            if loss_reduction[best_split] > 0: # Only keep split if loss is reduced
                                candidates.append((loss_0_1[best_split], loss_bce[best_split], l, f, best_split))
                        if plot: ax[l].plot(leaf.features_and_thresholds[f], loss_reduction, marker="o")
                if len(candidates) == 0: break # If loss reduction not possible, stop growth
                # Identify and make the best split across all leaves and features
                current_loss_0_1, current_loss_bce, l, f, best_split = sorted(candidates, key=lambda c: c[0 if loss_func == "0-1" else 1])[0]
                node = self.leaves[l]
                node.split(f, best_split)
                print(f"{len(self.leaves)}: Split leaf {l} at {f.__name__}={node.threshold} (loss_0_1={current_loss_0_1}, loss_bce={current_loss_bce})")
                # Update attributes for use in future growth
                self.leaves = self.leaves[:l] + [node.left, node.right] + self.leaves[l+1:]
                self.r_mean = torch.cat([self.r_mean[:l], node.r_mean[f][best_split], self.r_mean[l+1:]])
                idx_to_leaf[idx_to_leaf > l] += 1 # Increment leaf numbers
                idx_to_leaf[node.right.ind] += 1
                # Run callbacks
                if callbacks is not None:
                    for callback in callbacks:
                        callback(self)

            #####################
            # TODO: pruning stage
            #####################

    def to_hyperrectangles(self):
        """Convert tree into a form that enables visual and textual representation.
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
    """Class for a node in the tree."""
    def __init__(self, ind, features_and_thresholds):
        # Populated on creation
        self.ind, self.features_and_thresholds = ind, features_and_thresholds
        # Populated on split evaluation
        self.ind_right, self.r_mean = {}, {}
        # Populated on split acceptance via self.split()
        self.feature, self.threshold = None, None
        self.left, self.right = None, None

    def __call__(self, states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor) -> torch.Tensor:
        return self.feature(states, actions, next_states) < self.threshold

    def split(self, f, split):
        self.feature = f
        # Create two child nodes
        ind_right = self.ind_right[f][split]
        ind_left = self.ind[~torch.isin(self.ind, ind_right)]
        assert len(ind_left) + len(ind_right) == len(self.ind)
        self.left  = Node(ind_left,  self.features_and_thresholds.copy())
        self.right = Node(ind_right, self.features_and_thresholds.copy())
        # Simultaneously set split threshold for this node, and candidate thresholds for children
        self.left.features_and_thresholds[f], self.threshold, self.right.features_and_thresholds[f] = \
            torch.split(self.features_and_thresholds[f], [split, 1, len(self.features_and_thresholds[f])-split-1])
        self.threshold = self.threshold.squeeze()
