import math
from typing import Tuple, List

import numpy as np
from all.agents import Agent
from all.environments import Environment


def shared_autonomy_tabular(optimal_agent: Agent, human_agent: Agent, env: Environment, alpha: float, max_timesteps: int = 500) -> Tuple[int, float]:
    """executing shared autonomy (not actually training) for one episode

    Args:
        optimal_agent (Agent): agent with a "optimal" policy
            assumes it has a tabular (ndarray) Q function which can be accessed by Agent.Q
        human_agent (Agent): wrapper for a human policy
        env (Environment):
        alpha (float): range [0, 1], hyperparameter that controls the tolerance of the system to suboptimal human suggestions
        max_timesteps (int, optional): maximum amount of timesteps to execute the environment

    Returns:
        Tuple[int, float]: element 0 is the number of interventions made, element 1 is the reward achieved
    """
    q_optimal = optimal_agent.Q

    env.reset()

    returns = 0
    interventions = 0

    done = False
    timestep = 0
    while not done and (timestep < max_timesteps):
        # get the action of each agent
        human_action = human_agent.eval(env.state)
        optimal_action = optimal_agent.eval(env.state)

        # get q_values for each agent's action
        q_vals = q_optimal[env.state['observation'].numpy()[0]]
        # subtract away the minimum q value to make them comparable if some are negative
        q_vals -= min(q_vals)
        q_val_human = q_vals[human_action]
        q_val_optimal = q_vals[optimal_action]

        # deciding if we need to intervene, equation 3 of Shared Autonomy without similarity function
        if q_val_human > ((1 - alpha) * q_val_optimal):
            action = human_action
        else:
            action = optimal_action
            interventions += 1

        env.step(action)
        returns += env.state['reward']
        done = env.state['done']
        timestep += 1

    return interventions, returns


def shared_modified_tabular(optimal_agent: Agent, human_agent: Agent, env: Environment, alpha: float, max_timesteps: int = 500) -> Tuple[int, float]:
    """Executing 'shared autonomy modified' for one episode
    This method is from the work done by Aarshee Mishra, Abhik Singla, 
    Nicholas Perello, and William Goss from 2020's CS696DS course.

    Args:
        optimal_agent (Agent): agent with a "optimal" policy
            assumes it has a tabular (ndarray) Q function which can be accessed by Agent.Q
        human_agent (Agent): wrapper for a human policy
        env (Environment):
        alpha (float): range [0, 1], hyperparameter that controls the tolerance of the system to suboptimal human suggestions
        max_timesteps (int, optional): maximum amount of timesteps to execute the environment

    Returns:
        Tuple[int, float]: element 0 is the number of interventions made, element 1 is the reward achieved
    """
    q_optimal = optimal_agent.Q

    env.reset()

    returns = 0
    interventions = 0

    done = False
    timestep = 0
    while not done and (timestep < max_timesteps):
        # get the action of each agent
        human_action = human_agent.eval(env.state)
        optimal_action = optimal_agent.eval(env.state)

        # get q_values for each agent's action
        q_vals = q_optimal[env.state['observation'].numpy()[0]]
        # subtract away the minimum q value to make them comparable if some are negative
        q_vals -= min(q_vals)
        q_val_human = q_vals[human_action]
        q_val_optimal = q_vals[optimal_action]

        # deciding if we need to intervene, an interpretation of section 3.2.1 from previous work
        curr_gamma = math.pow(env._gamma, env._timestep)
        if (curr_gamma * q_val_human) + returns >= (1 - alpha) * q_val_optimal * curr_gamma:
            action = human_action
        else:
            action = optimal_action
            interventions += 1

        env.step(action)
        returns += env.state['reward']
        done = env.state['done']
        timestep += 1

    return interventions, returns


def evaluate_shared_autonomy_tabular(optimal_agent: Agent, human_agent: Agent, env: Environment, alpha: float, max_timesteps: int = 500, num_episodes: int = 1000, sharing_method: str = 'default') -> Tuple[List, List]:
    """wrapper for getting an estimate of the number of interventions and reward per episode

    Args:
        optimal_agent (Agent): agent with a "optimal" policy
            assumes it has a tabular (ndarray) Q function which can be accessed by Agent.Q
        human_agent (Agent): wrapper for a human policy
        env (Environment):
        alpha (float): range [0, 1], hyperparameter that controls the tolerance of the system to suboptimal human suggestions
        max_timesteps (int, optional): maximum amount of timesteps to execute the environment
        num_episodes (int, optional): the number of episodes to run shared autonomy from scratch
        sharing_method (str): 'default' to use shared_autonomy_tabular, and 'modified' to use shared_modified_tabular

    Returns:
        Tuple[List, List]: element 0 is a list of the number interventions in the corresponding episode, 
                           element 1 is a list of the reward achieved in the corresponding episode
    """

    if sharing_method == 'default':
        sharing_func = shared_autonomy_tabular
    elif sharing_method == 'modified':
        sharing_func = shared_modified_tabular
    else:
        raise ValueError('sharing_method {} is not a valid option'.format(sharing_method))

    total_interventions = []
    total_reward = []
    for _ in range(num_episodes):
        interventions, reward = sharing_func(optimal_agent, human_agent, env, alpha)
        total_interventions.append(interventions)
        total_reward.append(reward)

    return total_interventions, total_reward
