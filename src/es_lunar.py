"""Test script for Evolution Stategies method of training an RL agent

Reference ES implemention https://github.com/atgambardella/pytorch-es
"""
import math

import gym
import torch
import numpy as np

from environments.hitl_gym_environment import HITLGymEnvironment
from agents.evolution_strategy_agent import EvolutionStrategyAgent, SimplePolicyNetwork
from utils.evaluate_policies import evaluate_policy_once

# Env state: Appends human's action to the environment's state representation
# for now we assume the user always does nothing (so it essentially the normal BBO)
#
# Reward: game reward + penalize intervention (look at agent's action)
# for now we don't modify the reward
env = HITLGymEnvironment(env=gym.make('CartPole-v0'), human=None)

# agent which needs to map the state to an action
# also needs to be trained by being given the reward signal (for an entire episode)
agent = EvolutionStrategyAgent(env.theta_size, env.num_actions)


def fitness_shaping(returns):
    """ A rank transformation on the rewards, which reduces the chances of falling into local optima early in training.
    Essentially rescales the rewards to be small positive numbers.
    (This is taken from the reference implementation)
    """
    sorted_returns_backwards = sorted(returns)[::-1]
    lamb = len(returns)
    shaped_returns = []
    denom = sum([max(0, math.log(lamb / 2 + 1, 2) - math.log(sorted_returns_backwards.index(r) + 1, 2)) for r in returns])
    for r in returns:
        num = max(0, math.log(lamb / 2 + 1, 2) - math.log(sorted_returns_backwards.index(r) + 1, 2))
        shaped_returns.append(num / denom + 1 / lamb)
    return shaped_returns


# Evolution Strategies training algorithm --------------------

# timesteps to train the agent
TRAIN_ITERATIONS = 10000
# hyperparameter in evolution strategies which controls the number of "mutations" per training timestep
# note that this will technically result in a population twice the value because we consider the antithesis for each perturbation
POPULATION_SIZE = 40
# hyperparameter, noise standard deviation
SIGMA = 0.05
LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.999

N = POPULATION_SIZE * 2
for t in range(TRAIN_ITERATIONS):
    returns, was_anti, epsilon_hist = [], [], []
    for i in range(POPULATION_SIZE):
        # create two new candidate networks
        new_agent = EvolutionStrategyAgent(env.theta_size, env.num_actions)
        new_agent.policy = SimplePolicyNetwork(env.theta_size, env.num_actions)
        new_agent_anti = EvolutionStrategyAgent(env.theta_size, env.num_actions)
        new_agent_anti.policy = SimplePolicyNetwork(env.theta_size, env.num_actions)

        # initialize these new models to have the same parameters as the current agent
        new_agent.policy.load_state_dict(agent.policy.state_dict())
        new_agent_anti.policy.load_state_dict(agent.policy.state_dict())

        # for each type of parameter in the policy network
        epsilons = []
        for (_, v), (_, anti_v) in zip(new_agent.policy.get_params(), new_agent_anti.policy.get_params()):
            # sample isotropic multivariate Gaussians
            epsilon = np.random.normal(0, 1, v.size())
            # store each epsilon for computing the gradient
            epsilons.append(epsilon)
            # perturb the parameters according to the sampling
            v += torch.from_numpy(SIGMA * epsilon).float()
            anti_v -= torch.from_numpy(SIGMA * epsilon).float()
        epsilon_hist.append(epsilons)

        # compute returns for this perturbation
        ep_return = evaluate_policy_once(new_agent, env, max_timesteps=10000)
        ep_return_anti = evaluate_policy_once(new_agent_anti, env, max_timesteps=10000)
        # store the return for the gradient
        returns.extend([ep_return, ep_return_anti])
        # append if we need to multiply a piece of the gradient by -1 or not
        was_anti.extend([1, -1])

    # gradient update of the agent's policy parameters based on returns
    returns = fitness_shaping(returns)
    for i in range(POPULATION_SIZE):
        for idx, kv in enumerate(agent.policy.get_params()):
            _, v = kv
            update = (LEARNING_RATE / (N * SIGMA)) * (returns[i] * was_anti[i] * epsilon_hist[i][idx])
            v += torch.from_numpy(update)

    LEARNING_RATE *= LEARNING_RATE_DECAY

    print('Iteration', t, evaluate_policy_once(agent, env, max_timesteps=10000))
