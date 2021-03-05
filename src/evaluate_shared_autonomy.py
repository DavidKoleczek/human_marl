# Script to evaluate baseline shared autonomy methods on a simple Gridworld and two different simulated human policies
# Generates a plot showing the number of interventions and returns on different values of alpha

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from agents.qlearningtabular_agent import QLearningTabularAgent
from environments.gridworld_environment import GridworldEnvironment
from agents.train_agents import train_optimal_agent_tabularq
from utils.evaluate_policies import average_policy_returns
from agents.humantabular_agent import HumanTabularAgent
from sharedautonomy.baseline import evaluate_shared_autonomy_tabular

env = GridworldEnvironment()
agent = QLearningTabularAgent(action_space=list(range(4)), q_dims=(env._grid_dims[0] * env._grid_dims[1], 4))

optimal_agent = train_optimal_agent_tabularq(agent, env)
optimal_return = average_policy_returns(optimal_agent, env)
print(optimal_return)
print("finish learning")

# make sure we actually trained a good policy for Gridworld, sometimes it diverges?
assert optimal_return > 3

(u, d, l, r) = (0, 1, 2, 3)
human_policy_better = [
    r, r, r, r, d,
    u, u, u, r, d,
    u, u, r, r, d,
    u, u, r, r, d,
    u, u, r, r, d]
human_agent_better = HumanTabularAgent(human_policy_better)

human_policy_worse = [
    r, r, r, r, d,
    u, d, l, l, l,
    d, l, l, r, d,
    d, r, r, u, d,
    r, r, u, u, u]
human_agent_worse = HumanTabularAgent(human_policy_worse)


def evaluate_helper(human_agent, plot_name, env, optimal_agent):
    """Executes shared autonomy for various different values of alpha and saves a plot of the result
    """
    alphas = []
    intervention_history = []
    returns_history = []
    intervention_history_mod = []
    returns_history_mod = []
    for alpha in range(1, 98):
        print("alpha", alpha)
        # want alphas to be between 0 and 1
        alpha = alpha / 100

        # get results over different alphas for the default shared autonomy method
        interventions, returns = evaluate_shared_autonomy_tabular(optimal_agent, human_agent, env, alpha=alpha, sharing_method='default')
        alphas.append(alpha)
        intervention_history.append(np.average(interventions))
        returns_history.append(np.average(returns))

        # get results over different alphas for the modified shared autonomy method
        interventions, returns = evaluate_shared_autonomy_tabular(optimal_agent, human_agent, env, alpha=alpha, sharing_method='modified')
        intervention_history_mod.append(np.average(interventions))
        returns_history_mod.append(np.average(returns))

    plot_data = pd.DataFrame({
        'Alpha': alphas,
        'Default Interventions': intervention_history,
        'Default Returns': returns_history,
        'Modified Interventions': intervention_history_mod,
        'Modified Returns': returns_history_mod
    }).melt(id_vars='Alpha', var_name='Type')
    sns.lineplot(data=plot_data, x='Alpha', y='value', hue='Type')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(plot_name)
    plt.clf()  # clear matplotlib so we don't get plots on top of each other


evaluate_helper(human_agent_better, 'plots/compare_baselines_better_human.png', env, optimal_agent)
evaluate_helper(human_agent_worse, 'plots/compare_baselines_worse_human.png', env, optimal_agent)
