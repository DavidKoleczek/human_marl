""" Human in the loop wrapper for GymEnvironments """
from copy import deepcopy

import gym
import torch
import numpy as np
from gym.spaces import Space, Box, Discrete

from easyrl.environments import GridworldEnvironment
from easyrl.core import State
from easyrl.agents import Agent


class HITLGridworldEnvironment(GridworldEnvironment):
    """ Human in the loop wrapper for GymEnvironments
    """

    def __init__(self, human: Agent):
        """Initializes a human in the loop wrapper around a GymEnvironment

        Args:
            human (TODO): TODO human should be a simulated agent that given the current state of the GymEnvironment will return some action
            device (str, optional): Defaults to torch.device('cpu').
        """
        super().__init__()
        self.human = human

    def reset(self) -> State:
        """
        Reset the environment and return a new intial state.

        Returns
        -------
        State
            The initial state for the next episode.
        """
        self._state = State({
            # for now assume the human starts with the 0 action
            'observation': torch.cat((self._start_state, torch.tensor([0])), 0),
            'reward': 0,
            'done': False
        })
        self._reward = 0
        self._action = None
        self._timestep = 0
        self._done = False
        return self._state

    def step(self, action) -> State:
        """Overrides GymEnvironment's step method. Returns the state of the gym environment + the human's action.
        Modifies the reward depending on if the provided action constitutes an intervention

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """
        # TODO: Modify reward
        # TODO: Add noop to agent
        state = super().step(action)
        human_action = self.human.eval(state)
        state['observation'] = torch.cat((state['observation'], torch.tensor([human_action])), 0)
        return state

    @property
    def state(self):
        return self._state

    @property
    def state_space(self) -> Space:
        """
        The Space representing the range of observable states.

        Returns
        -------
        Space
            An object of type Space that represents possible states the agent may observe
        """
        shape = (self._grid_dims[0] * self._grid_dims[1]) + 1
        low = np.zeros(shape=shape, dtype=np.float32)
        high = np.ones(shape=shape, dtype=np.float32)
        return Box(low=low, high=high, shape=(shape, ), dtype=np.float32)

    @property
    def observation_space(self) -> Space:
        """
        Alias for Environemnt.state_space.

        Returns
        -------
        Space
            An object of type Space that represents possible states the agent may observe
        """
        shape = (self._grid_dims[0] * self._grid_dims[1]) + 1
        low = np.zeros(shape=shape, dtype=np.float32)
        high = np.ones(shape=shape, dtype=np.float32)
        return Box(low=low, high=high, shape=(shape, ), dtype=np.float32)

    @property
    def num_actions(self) -> int:
        """get the number of actions for use in the Policy Network

        Returns:
            int: number of actions
        """
        # check if the gym has a discrete state space
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.action_space.n

        raise ValueError('ES Agent currently only supports Discrete action spaces.')
