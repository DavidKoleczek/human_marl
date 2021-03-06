import random
from typing import Tuple, List

import numpy as np

from all.agents import Agent
from all.core import State


class QLearningTabularAgent(Agent):
    """Trains a tabular Q-Learning Agent

    Attributes:
        Q: a ndarray representing the agent's Q function
    """

    def __init__(self, action_space: List[int], q_dims: Tuple[int, int], init_q_val: float = 10, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.05):
        """Inits class with variables needed for training

        Args:
            action_space (List[int]): a list of discrete actions the agent can take
            q_dims (Tuple[int, int]): dimension of the q function (states x actions)
            init_q_val (float, optional): [description]. value to initialize the q funtion to
            alpha (float, optional): learning rate
            gamma (float, optional): 
            epsilon (float, optional): "exploration" threshold for epsilon greedy action selection
        """
        self.Q = np.full(shape=q_dims, fill_value=init_q_val, dtype=np.float64)
        self._action_space = action_space
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

    def _epsilon_greedy_action_selection(self, state: State) -> int:
        """Applies epsilon greedy action selection to choose the next action given the current Q function and state

        With probability <= epsilon this will chose a complete random action,
        otherwise it will chose the action with the highest q value for the state

        Args:
            state (State): current state of the Environment

        Returns:
            int: the next action
        """
        state = int(state['observation'].numpy()[0])
        epsilon = random.uniform(0, 1)
        if epsilon > self._epsilon:
            # choose one of the actions with the highest Q-values (breaking ties between maximum Q values)
            action = np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))
        else:
            # epsilon of the time choose a completely random action
            action = np.random.choice(self._action_space)

        return action

    def act(self, state: State, action: int, next_state: State) -> int:
        """Performs one training step and returns the next action
        """
        state = int(state['observation'].numpy()[0])
        reward = next_state['reward']
        next_state = int(next_state['observation'].numpy()[0])

        # q-learning update
        next_action = np.argmax(self.Q[next_state])
        update = self._alpha * (reward + (self._gamma * (self.Q[next_state, next_action] - self.Q[state, action])))
        self.Q[state, action] += update

        return next_action

    def eval(self, state: State) -> int:
        """Chooses an action using the current policy (Q function) of the agent
        """
        # always choose the greedy action when evaluating
        state = int(state['observation'].numpy()[0])
        return np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))
