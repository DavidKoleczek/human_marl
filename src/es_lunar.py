"""Test script for Evolution Stategies method of training an RL agent
"""
import gym
from environments.hitl_gym_environment import HITLGymEnvironment

from agents.evolution_strategy_agent import EvolutionStrategyAgent
from utils.evaluate_policies import evaluate_policy_once

# Env state: Appends human's action to the environment's state representation
#   for now we assume the user always does nothing (so it essentially the normal BBO)
# Reward: game reward + penalize intervention (look at agent's action)
#   for now we don't modify the reward
env = HITLGymEnvironment(env=gym.make('LunarLander-v2'), human=None)

# agent which needs to map the state to an action
# also needs to be trained by being given the reward signal (for an entire episode)
agent = EvolutionStrategyAgent(env.action_space)

# Evolution Strategies training algorithm --------------------
# timesteps to train the agent
train_iterations = 3
# hyperparameter in evolution strategies which controls the number of "mutations" per training timestep
population_size = 10
for t in range(train_iterations):
    returns = []
    for i in range(population_size):
        # TODO: sample isotropic multivariate Gaussians
        # TODO: compute returns for this perturbation
        # this line is related to the above todo, but need to modify agent's params first
        ep_return = evaluate_policy_once(agent, env, max_timesteps=10000)

    # TODO: gradient update of the agent's policy parameters based on returns
