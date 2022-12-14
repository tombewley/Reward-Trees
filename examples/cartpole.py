"""
Example script for reward tree learning in CartPole.
"""

from reward_trees import RewardTree, RewardNet
from argparse import ArgumentParser
import gymnasium as gym
import torch
from numpy import array, pi
from random import Random
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument("--generator",       type=str, default="sample",     choices={"sample", "rollout"})
parser.add_argument("--feedback_mode",   type=str, default="preference", choices={"preference", "annotation"})
parser.add_argument("--num_eps",         type=int, default=500 )
parser.add_argument("--ep_length",       type=int, default=5   )  # NOTE: Need around 50 if generator=rollout
parser.add_argument("--num_preferences", type=int, default=1500)
parser.add_argument("--num_leaves",      type=int, default=5   )
parser.add_argument("--seed",            type=int, default=0   )
args = parser.parse_args()

# Define features; here we're just using the raw environment state dimensions
# NOTE: these use the next state in each state/action/next state tuple
def x        (s, a, ns): return ns[...,0]
def x_dot    (s, a, ns): return ns[...,1]
def theta    (s, a, ns): return ns[...,2]
def theta_dot(s, a, ns): return ns[...,3]

# Define ground-truth 'oracle' reward function, as used in the original environment
def terminated(s, a, ns):
    x_, theta_ = x(s, a, ns), theta(s, a, ns)
    theta_threshold_radians = 12 * 2 * pi / 360
    return torch.logical_or(
        torch.logical_or(x_ < -2.4, x_ > 2.4),
        torch.logical_or(theta_ < -theta_threshold_radians, theta_ > theta_threshold_radians)
    )
def not_terminated(s, a, ns): return ~terminated(s, a, ns)
def oracle(s, a, ns): return (not_terminated(s, a, ns)).sum()

# Load up CartPole, then add bounds to the state space (infinite by default)
cartpole = gym.make("CartPole-v1")
cartpole.observation_space.low [1] = -2
cartpole.observation_space.low [3] = -3
cartpole.observation_space.high[1] = +2
cartpole.observation_space.high[3] = +3
lo, hi = cartpole.observation_space.low, cartpole.observation_space.high

# Initialise the reward tree, providing the set of features and split thresholds within the bounds
tree = RewardTree(
    features_and_thresholds={
        x         : torch.arange(lo[0], hi[0], 0.05 ),
        x_dot     : torch.arange(lo[1], hi[1], 0.05 ),
        theta     : torch.arange(lo[2], hi[2], 0.005),
        theta_dot : torch.arange(lo[3], hi[3], 0.05 )
    },
    embed_by_ep=args.feedback_mode == "preference",
    max_num_eps=args.num_eps, max_ep_length=args.ep_length, seed=args.seed
)

# Also initialise a neural network model for comparison
net = RewardNet(features=(x, x_dot, theta, theta_dot), seed=args.seed)

n = args.num_eps * args.ep_length
cartpole.action_space.seed(args.seed)
if args.generator == "sample":
    # Generate 'pseudo-episodes' by randomly sampling from the bounded state space
    cartpole.observation_space.seed(args.seed)
    states  = torch.tensor(array([cartpole.observation_space.sample() for _ in range(n+1)]), device=tree.device)
    actions = torch.tensor(array([cartpole.action_space.sample() for _ in range(n)]), device=tree.device)
    ep_nums = torch.arange(args.num_eps, device=tree.device).repeat_interleave(args.ep_length)
    next_states = states[1:]
    states = states[:-1]
elif args.generator == "rollout":
    # Generate episodes by rolling out a random policy
    states, actions, next_states, ep_nums = [], [], [], []
    ep = -1
    for t in range(n):
        if t % args.ep_length == 0:
            ep += 1
            s, _ = cartpole.reset(seed=args.seed+ep) # Use episode number to increment seed
        a = cartpole.action_space.sample()
        ns, _, _, _, _ = cartpole.step(a)
        states.append(s); actions.append(a); next_states.append(ns); ep_nums.append(ep)
        s = ns
    states = torch.tensor(array(states), device=tree.device)
    actions = torch.tensor(array(actions), device=tree.device)
    next_states = torch.tensor(array(next_states), device=tree.device)
    ep_nums = torch.tensor(array(ep_nums), device=tree.device)

# Add these episodes to both models
tree.add_transitions(states, actions, next_states, ep_nums)
net .add_transitions(states, actions, next_states, ep_nums)

# Generate a bunch of preferences over randomly sampled pairs of episodes
# Preferences are deterministic; the one with higher oracle return is always preferred
rng = Random(args.seed)
tried_already = set()
while len(tree.preferences) < args.num_preferences:
    i = rng.randint(0, args.num_eps-1)
    ind_i = tree.get_indices(i)
    if args.feedback_mode == "preference":
        # Pairwise preferences over two episodes
        j = rng.randint(0, i)
        if (i, j) not in tried_already:
            ind_j = tree.get_indices(j)
            diff = oracle(tree.states[ind_i], tree.actions[ind_i], tree.next_states[ind_i]) \
                 - oracle(tree.states[ind_j], tree.actions[ind_j], tree.next_states[ind_j])
            # Discard cases of equal return
            # NOTE: might not always want to do this, but it's necessary if using loss_func="0-1"
            if diff:
                preference = (0. if diff < 0. else 1.)
                tree.add_preference(i, j, preference)
                net .add_preference(i, j, preference)
            tried_already.add((i, j))
    elif args.feedback_mode == "annotation":
        # Positive/negative annotation of a single episode
        if i not in tried_already:
            annotation = not_terminated(tree.states[ind_i], tree.actions[ind_i], tree.next_states[ind_i]).int()
            tree.add_annotation(i, annotation)
            net .add_annotation(i, annotation)
            tried_already.add(i)

# Train both models on the preference dataset
tree.train(max_num_leaves=args.num_leaves, num_batches=2000, loss_func="bce")
net .train(num_batches=2000)

# Predict rewards for all transitions in the dataset
s, a, ns = tree.states, tree.actions, tree.next_states
tree_rewards = tree(s, a, ns).detach().numpy()
net_rewards  = net (s, a, ns).detach().numpy()

fig, ax = plt.subplots(1, 2)
plot_features = x, theta
f0, f1 = (net.model.features.index(f) for f in plot_features)
if False:
    # Requires https://github.com/tombewley/hyperrectangles
    from hyperrectangles import show_rectangles
    lo, hi = ns.min(dim=0)[0], ns.max(dim=0)[0]
    show_rectangles(tree.to_hyperrectangles(), ax=ax[0],
        vis_dims=(f0, f1),
        # attribute=("mean", "reward"),
        vis_lims=((lo[f0], hi[f0]), (lo[f1], hi[f1])),
        cmap_lims=(tree_rewards.min(), tree_rewards.max()),
        edge_colour="k",
        cbar=False
    )

# Scatter plot reward predictions
ax[0].scatter(ns[:,f0], ns[:,f1], c=tree_rewards, s=3, cmap="coolwarm_r", zorder=-2)
ax[1].scatter(ns[:,f0], ns[:,f1], c=net_rewards,  s=3, cmap="coolwarm_r", zorder=-2)
ax[0].set_xlabel(plot_features[0].__name__); ax[0].set_ylabel(plot_features[1].__name__)
plt.suptitle("Reward predictions for CartPole")
ax[0].set_title("Tree"); ax[1].set_title("Net")

plt.show()
