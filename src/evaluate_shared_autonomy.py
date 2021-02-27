import numpy as np

from agents.qlearningtabular_agent import QLearningTabularAgent
from environments.gridworld_environment import GridworldEnvironment
from agents.train_agents import train_optimal_agent
from utils.evaluate_policies import average_policy_returns
from agents.humantabular_agent import HumanTabularAgent
from sharedautonomy.baseline import evaluate_shared_autonomy_tabular

env = GridworldEnvironment()
agent = QLearningTabularAgent(action_space=list(range(4)), q_dims=(env._grid_dims[0] * env._grid_dims[1], 4))

optimal_agent = train_optimal_agent(agent, env)
optimal_return = average_policy_returns(agent, env)

# make sure we actually trained a good policy for Gridworld, sometimes it diverges?
assert optimal_return > 3

(u, d, l, r) = (0, 1, 2, 3)
human_policy = [
    r, r, r, r, d,
    u, u, u, r, d,
    u, u, r, r, d,
    u, u, r, r, d,
    u, u, r, r, d]
human_agent = HumanTabularAgent(human_policy)

interventions, returns = evaluate_shared_autonomy_tabular(optimal_agent, human_agent, env, alpha=0.1)

print('Average Interventions: {}, Average Reward: {}'.format(np.average(interventions), np.average(returns)))
