import torch
from typing import Union
from collections import namedtuple
from random import Random


preference_tuple = namedtuple("preference_tuple", ("i", "j", "y", "w", "info"))

class RewardLearner:
    """Base class for a preference-based reward learner."""
    sigmoid = torch.nn.Sigmoid()
    bce_loss = torch.nn.BCELoss()
    bce_loss_noreduce = torch.nn.BCELoss(reduction="none")

    def __init__(self, model, integrator:str="mean", embed_by_ep:bool=False, negative_rewards:bool=False, seed:int=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.integrator = integrator
        self.embed_by_ep = embed_by_ep
        self.negative_rewards = negative_rewards
        self.states, self.actions, self.next_states, self.ep_nums = None, None, None, None
        self.preferences = []
        self.shift, self.scale = 0., 1.
        self.seed(seed)

    def seed(self, seed:int):
        """Seed the random number generator."""
        self.rng = Random(seed)

    def __call__(self, states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor) -> torch.Tensor:
        """Given tensors of transitions, predict rewards."""
        if issubclass(self.model.__class__, torch.nn.Embedding): raise Exception("Cannot call for unseen transitions")
        else: return (self.model(states, actions, next_states) - self.shift) / self.scale

    def add_transitions(self, states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor, ep_nums:torch.Tensor):
        """Add transitions to the dataset."""
        assert states.device == actions.device == next_states.device == ep_nums.device == self.device
        assert states.shape[0] == actions.shape[0] == next_states.shape[0] == ep_nums.shape[0]
        assert states.shape == next_states.shape
        assert ep_nums.dim() == 1
        if self.states is None:
            self.states, self.actions, self.next_states, self.ep_nums = states, actions, next_states, ep_nums
        else:
            assert states.shape[1:] == self.states.shape[1:]
            assert actions.shape[1:] == self.actions.shape[1:]
            self.states      = torch.cat((self.states, states))
            self.actions     = torch.cat((self.actions, actions))
            self.next_states = torch.cat((self.next_states, next_states))
            self.ep_nums     = torch.cat((self.ep_nums, ep_nums))

    def add_preference(self, i:Union[torch.Tensor, int, callable], j:Union[torch.Tensor, int, callable], preference:float, weight:float=1., **info):
        """Add a preference to the dataset (1 means i preferred, 0 means j preferred).
        i and j are either tensors specifying indices in the dataset, integers specifying ep_nums, or binary selection functions.
        """
        assert (type(i) in {torch.Tensor, int} or callable(i)) and (type(j) in {torch.Tensor, int} or callable(j)), f"Invalid type for i or j"
        assert type(preference) == float and 0. <= preference <= 1., f"Invalid preference value: {preference}"
        assert type(weight) == float and 0. < weight, f"Invalid weight value: {weight}"
        self.preferences.append(preference_tuple(i, j, preference, weight, info))

    def update_on_batch(self, batch_size:int):
        """Sample a random batch of preferences and update the model using torch.nn.BCELoss."""
        k = len(self.preferences)
        if k == 0: print("No preference data!"); return
        # Sample a preference batch with replacement according to weights
        batch = [(self.get_indices(p.i), self.get_indices(p.j), p.y) for p in
                  self.rng.choices(self.preferences, weights=[p.w for p in self.preferences], k=min(batch_size, k))]
        # If the same transition is present in multiple preferences, only need to predict its reward once
        rewards = torch.empty(self.states.shape[0], device=self.device)
        batch_indices = torch.unique(torch.cat([torch.cat([i, j]) for i, j, _ in batch]))
        rewards[batch_indices] = self.indices_to_rewards(batch_indices)
        # Preference targets
        y_batch = torch.tensor([y for _, _, y in batch], device=self.device)
        # Preference predictions (via Bradley-Terry model)
        y_pred = torch.cat([self.sigmoid(self.integrate_utility(rewards[i]) - self.integrate_utility(rewards[j])
                                         ).unsqueeze(0) for i, j, _ in batch])
        # Minimise BCE loss
        loss = self.bce_loss(y_pred, y_batch)
        self.model.optimise(loss)
        return loss

    def integrate_utility(self, rewards:torch.Tensor):
        """Integrate a vector of rewards into a scalar utility value."""
        if self.integrator == "sum":
            return rewards.sum(dim=-1)
        elif self.integrator == "mean":
            # NOTE: this handles the  "null" zero-reward case
            return rewards.mean(dim=-1) if rewards.shape[-1] > 0 else torch.zeros(rewards.shape[:-1], device=self.device)
        else:
            raise NotImplementedError

    def normalise(self):
        """Normalise rewards to have unit standard deviation on the training set, with a common sign (+/-)."""
        with torch.no_grad():
            rewards = self.indices_to_rewards()
            self.shift = rewards.max() if self.negative_rewards else rewards.min()
            self.scale = rewards.std()

    def get_indices(self, i:Union[torch.Tensor, int, callable]) -> torch.Tensor:
        """If i is a tensor, return it unchanged.
        If i is an integer, interpret it as an ep_num and look up indices in the dataset.
        If i is a function, run that function to get indices.
        """
        if type(i) == int: return torch.nonzero(self.ep_nums == i).squeeze(1)
        elif callable(i):  return torch.nonzero(i(self.states, self.actions, self.next_states)).squeeze(1)
        else: return i

    def indices_to_rewards(self, i:torch.Tensor=None) -> torch.Tensor:
        """Given a tensor of indices, return rewards."""
        if i is None: i = torch.arange(len(self.states), device=self.device)
        # For embedding models, look up reward predictions by ep_num or index
        if issubclass(self.model.__class__, torch.nn.Embedding):
            rewards = self.model(self.ep_nums[i] if self.embed_by_ep else i)
        # For transition space models, use the transition data themselves
        else: rewards = self.model(self.states[i], self.actions[i], self.next_states[i])
        if torch.isnan(rewards).any(): raise Exception
        return rewards

    """Methods for grounding non-standard feedback into the canonical pairwise preference representation."""

    def add_annotation(self, ep_num:int, annotation:torch.Tensor, blank_weight:float=1.):
        """Add positive/negative annotations to an episode and parse into a set of inter-temporal preferences,
        as in "Improving Multimodal Interactive Agents with Reinforcement Learning from Human Feedback".
        For a length-T episode, the maximum number of preferences produced is T(T-1)/2.
        """
        ind = self.get_indices(ep_num)
        assert ind.shape == annotation.shape, "Annotation shape mismatch"
        for t2 in range(len(ind)+1):
            for t1 in range(t2):
                unique = annotation[t1:t2].unique()
                unique = unique[unique != 0]
                if len(unique) <= 1:
                    # A preference is added if all annotations in the time window have the same sign,
                    # or if there are no annotations at all
                    sign = unique.item() if len(unique) == 1 else 0
                    self.add_preference(
                        i=ind[t1:t2],
                        # In the fully-observable case, the original method is equivalent to taking preferences
                        # between time windows and a "null" zero-reward episode
                        j=torch.tensor([], dtype=int),
                        # NOTE: the interptetation of the blank case as an equal preference differs from the original
                        preference=(sign + 1.) / 2.,
                        weight=blank_weight if sign == 0 else 1.
                    )
