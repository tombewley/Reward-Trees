from . import RewardLearner
import torch


class RewardNet(RewardLearner):
    """Preference-based reward learner with a neural network model."""
    def __init__(self, features, seed:int=None):
        if seed is not None: torch.manual_seed(seed) # Need to do this here for seeded model initialisation
        RewardLearner.__init__(self, model=NetModel(features=features), seed=seed)

    def train(self, num_batches=500, batch_size=32):
        for i in range(num_batches): print(i, self.update_on_batch(batch_size).item())
        self.normalise()


class NetModel(torch.nn.Module):
    def __init__(self, features,
                 # 3x 256 hidden units, leaky ReLU, Adam with lr=3e-4 used in PEBBLE paper
                 hidden_layers=[256, 256, 256],
                 activation=torch.nn.LeakyReLU(),
                 optimiser=torch.optim.Adam,
                 lr=3e-4
                 ):
        super(NetModel, self).__init__()
        self.features = features
        layer_sizes = [len(self.features)] + hidden_layers + [1]
        layers = []
        for l in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[l], layer_sizes[l+1]))
            if l < len(layer_sizes) - 2: layers.append(activation)
        self.layers = torch.nn.Sequential(*layers)
        self.optimiser = optimiser(self.parameters(), lr=lr)

    def forward(self, states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([f(states, actions, next_states).unsqueeze(-1) for f in self.features], dim=-1)).squeeze()

    def optimise(self, loss): 
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
