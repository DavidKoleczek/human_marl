import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from agents.vac_agent import VAC_agent
from agents.qlearningtabular_agent import QLearningTabularAgent
from environments.gridworld_environment import GridworldEnvironment
from agents.train_agents import train_optimal_agent_tabularq
from utils.evaluate_policies import average_policy_returns
from agents.humantabular_agent import HumanTabularAgent
from sharedautonomy.baseline import evaluate_shared_autonomy_tabular

from environments.gridworld_environment import GridworldEnvironment  # pylint: disable=import-error

from utils.run_experiment import run_experiment

env = GridworldEnvironment(grid_dims = (3, 3), start_state = 0, end_states = [8], obstacle_states = [], water_states = [3])
#env = GridworldEnvironment(grid_dims = (5, 5), start_state = 0, end_states = [24], obstacle_states = [12, 17], water_states = [22])
frames = 20000
max_steps = 200

agent = VAC_agent(device = "cpu", clip_grad = 1, value_loss_scaling = 1)
run_experiment(agent, env, frames=frames)
