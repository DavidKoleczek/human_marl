import math
from typing import Tuple, List

import torch
import numpy as np
from gym.spaces import Space, Box, Discrete

from ._state import State
from ._environment import Environment


class GridworldEnvironment(Environment):
    def __init__(self,
                 grid_dims: Tuple[int, int] = (5, 5),
                 start_state: int = 0,
                 end_states: List[int] = [24],
                 obstacle_states: List[int] = [12, 17],
                 water_states: List[int] = [22]):
        """
        A Gridworld based on the one described in the lecture notes of the 687 course material.
        Found here https://people.cs.umass.edu/~pthomas/courses/CMPSCI_687_Fall2020/687_F20.pdf
        See page 10 for a pictorial representation and more details.
        It is modified to have diagonal actions.

        Actions: up (0), down (1), left (2), right (3), up-left (4), up-right (5), down-left (6), down-right (7)

        Environment Dynamics:
            With probability 0.8 the robot moves in the specified direction.

            With probability 0.05 it gets confused and veers to the
            right -- it moves +90 degrees from where it attempted to move, e.g.,
            with probability 0.05, moving up will result in the robot moving right.

            With probability 0.05 it gets confused and veers to the left -- moves
            -90 degrees from where it attempted to move, e.g., with probability
            0.05, moving right will result in the robot moving down.

            With probability 0.1 the robot temporarily breaks and does not move at all.

            If the movement defined by these dynamics would cause the agent to
            exit the grid (e.g., move up when next to the top wall), then the
            agent does not move.

            The robot starts in the top left corner, and the process ends in the bottom right corner.
            When the action is do nothing(4) robot always stays at the same place.

        Rewards: -10 for entering the state with water
                +10 for entering the goal state
                0 everywhere else
        """

        # customizable parameters
        self._grid_dims = grid_dims
        self._start_state = self._one_hot_state(start_state)
        self._end_states = end_states
        self._obstacle_states = obstacle_states
        self._water_states = water_states

        # fixed parameters
        self._gamma = 0.9
        # stochasticity
        self._pr_stay = 0.1
        self._pr_rotate = 0.05
        # dicts mapping actions to the appropriate rotations. it helps to draw this out if you are confused
        self._rotate_left = {0: 4, 1: 7, 2: 6, 3: 5, 4: 2, 5: 0, 6: 1, 7: 3}
        self._rotate_right = {0: 5, 1: 6, 2: 4, 3: 7, 4: 0, 5: 3, 6: 2, 7: 1}

        # set environment state
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._device = 'cpu'
        self._timestemp = 0
        self.reset()

    @property
    def name(self) -> str:
        """
        The name of the environment.
        """
        return 'gridworld_cs687_v0'

    def reset(self) -> None:
        """
        Reset the environment and return a new intial state.

        Returns
        -------
        State
            The initial state for the next episode.
        """
        # environment "state"
        self._state = State({
            'observation': self._start_state,  # pylint: disable=not-callable
            'reward': 0,
            'done': False
        })
        self._reward = 0
        self._action = None
        self._timestep = 0
        self._done = False
        return self._state

    def _calc_next_state(self, state: int, action: int) -> torch:
        """Calculates the next state given the current state, action, and environment dynamics

        Returns:
            int: the next state
        """

        if state in self._end_states:
            return state

        noise = np.random.uniform()
        if torch.is_tensor(action):
            action = action.item()
        if noise < self._pr_stay:  # do nothing
            return state
        elif noise < (self._pr_stay + self._pr_rotate):
            action = self._rotate_left[action]
        elif noise < (self._pr_stay + 2 * self._pr_rotate):
            action = self._rotate_right[action]

        # simulate taking a step in the environment
        next_state = state
        if action == 0:  # move up
            next_state = state - self._grid_dims[1]
        elif action == 1:  # move down
            next_state = state + self._grid_dims[1]
        elif action == 2 and (next_state % self._grid_dims[1] != 0):  # move left
            next_state = state - 1
        elif action == 3 and ((next_state + 1) % self._grid_dims[1] != 0):  # move right
            next_state = state + 1
        elif action == 4 and (next_state % self._grid_dims[1] != 0):  # move diagonal up-left
            next_state = state - self._grid_dims[1] - 1
        elif action == 5 and ((next_state + 1) % self._grid_dims[1] != 0):  # move diagonal up-right
            next_state = state - self._grid_dims[1] + 1
        elif action == 6 and (next_state % self._grid_dims[1] != 0):  # move diagonal down-left
            next_state = state + self._grid_dims[1] - 1
        elif action == 7 and ((next_state + 1) % self._grid_dims[1] != 0):  # move diagonal down-right
            next_state = state + self._grid_dims[1] + 1

        # check if the next state is valid and not an obstacle
        size = self._grid_dims[0] * self._grid_dims[1]
        if next_state >= 0 and next_state < size and next_state not in self._obstacle_states:
            return next_state
        else:
            return state

    def _one_hot_state(self, state: int) -> torch.tensor:
        one_hot_state = np.zeros(shape=(self._grid_dims[0] * self._grid_dims[1]), dtype=np.float32)
        one_hot_state[int(state)] = 1
        one_hot_state = torch.tensor(one_hot_state, dtype=torch.float32)
        return one_hot_state

    def _un_one_hot_state(self, state: torch.tensor) -> int:
        state = torch.argmax(state).numpy()
        state = int(state)
        return state

    def _calc_reward(self, state: int) -> float:
        """Calculates the reward for entering the given state
        """
        reward = 0.0
        if state in self._water_states:
            reward = -10 * math.pow(self._gamma, self._timestep)
        elif state in self._end_states:
            reward = 10 * math.pow(self._gamma, self._timestep)
        else:
            reward = 0.0
        return float(reward)

    def step(self, action: int) -> State:
        """
        Apply an action and get the next state.

        Parameters
        ----------
        action : Action
            The action to apply at the current time step.

        Returns
        -------
        all.environments.State
            The State of the environment after the action is applied.
            This State object includes both the done flag and any additional "info"
        float
            The reward achieved by the previous action
        """
        curr_state = self._un_one_hot_state(self._state['observation'])

        next_state = self._calc_next_state(curr_state, action)
        reward = self._calc_reward(next_state)

        self._reward = reward

        self._timestep += 1

        if self._timestep >= 200:
            self._done = True
        else:
            self._done = next_state in self._end_states

        next_state = self._one_hot_state(next_state)

        state_obj = State({
            'observation': next_state,
            'reward': reward,
            'done': self._done
        })
        self._state = state_obj

        return state_obj

    @property
    def state(self) -> State:
        """
        The State of the Environment at the current timestep.
        """
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
        shape = self._grid_dims[0] * self._grid_dims[1]
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
        shape = self._grid_dims[0] * self._grid_dims[1]
        low = np.zeros(shape=shape, dtype=np.float32)
        high = np.ones(shape=shape, dtype=np.float32)
        return Box(low=low, high=high, shape=(shape, ), dtype=np.float32)

    @property
    def action_space(self) -> Space:
        """
        The Space representing the range of possible actions.

        Returns
        -------
        Space
            An object of type Space that represents possible actions the agent may take
        """
        return Discrete(8)

    def render(self, **kwargs):
        """
        Render the current environment state.
        """
        return None

    def close(self):
        """
        Clean up any extraneaous environment objects.
        """
        return None

    def duplicate(self, n):
        """
        Create n copies of this environment.
        """
        return [GridworldEnvironment() for _ in range(n)]

    @property
    def device(self):
        """
        The torch device the environment lives on.
        """
        return self._device
