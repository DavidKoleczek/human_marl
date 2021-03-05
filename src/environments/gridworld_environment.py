import math
from typing import Tuple, List

import torch
import numpy as np
from all.environments import Environment
from all.core import State
from gym import spaces


class GridworldEnvironment(Environment):
    def __init__(self, grid_dims: Tuple[int, int] = (5, 5), start_state: int = 0, end_states: List[int] = [24], obstacle_states: List[int] = [12, 17], water_states: List[int] = [22]):
        """
        The Gridworld as described in the lecture notes of the 687 course material.
        Found here https://people.cs.umass.edu/~pthomas/courses/CMPSCI_687_Fall2020/687_F20.pdf
        See page 10 for a pictorial representation and more details

        Actions: up (0), down (1), left (2), right (3)

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
        self._start_state = start_state
        self._end_states = end_states
        self._obstacle_states = obstacle_states
        self._water_states = water_states

        # fixed parameters
        self._gamma = 0.99
        # stochasticity
        self._prStay = 0.1
        self._prRotate = 0.05
        # dicts mapping actions to the appropriate rotations
        self._rotateLeft = {0: 2, 1: 3, 2: 1, 3: 0}
        self._rotateRight = {0: 3, 1: 2, 2: 0, 3: 1}

        # set environment state
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
            'observation': torch.tensor([self._start_state]),  # pylint: disable=not-callable
            'reward': 0,
            'done': False
        })
        self._reward = 0
        self._action = None
        self._timestep = 0
        self._done = False

    def _calc_next_state(self, state: int, action: int) -> int:
        """Calculates the next state given the current state, action, and environment dynamics

        Returns:
            int: the next state
        """
        if state in self._end_states:
            return state

        noise = np.random.uniform()
        if torch.is_tensor(action):
            action = action.item()
        if noise < self._prStay:  # do nothing
            return state
        elif noise < (self._prStay + self._prRotate):
            action = self._rotateLeft[action]
        elif noise < (self._prStay + 2 * self._prRotate):
            action = self._rotateRight[action]

        # simulate taking a step in the environment
        nextState = state
        if action == 0:  # move up
            nextState = state - self._grid_dims[1]
        elif action == 1:  # move down
            nextState = state + self._grid_dims[1]
        elif action == 2 and (nextState % self._grid_dims[1] != 0):  # move left
            nextState = state - 1
        elif action == 3 and ((nextState + 1) % self._grid_dims[1] != 0):  # move right
            nextState = state + 1

        # check if the next state is valid and not an obstacle
        size = self._grid_dims[0] * self._grid_dims[1]
        if nextState >= 0 and nextState < size and nextState not in self._obstacle_states:
            return nextState
        else:
            return state

    def _calc_reward(self, state: int) -> float:
        """Calculates the reward for entering the given state
        """
        if state in self._water_states:
            return -10 * math.pow(self._gamma, self._timestep)
        elif state in self._end_states:
            return 10 * math.pow(self._gamma, self._timestep)
        else:
            return 0

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
        next_state = self._calc_next_state(self._state['observation'].numpy()[0], action)
        reward = self._calc_reward(next_state)

        self._reward = reward
        self._done = next_state in self._end_states
        self._timestep += 1

        state_obj = State({
            'observation': torch.tensor([next_state]),  # pylint: disable=not-callable
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
    def state_space(self) -> np.ndarray:
        """
        The Space representing the range of observable states.

        Returns
        -------
        Space
            An object of type Space that represents possible states the agent may observe
        """
        return np.arange(self._grid_dims[0] * self._grid_dims[1])

    @property
    def observation_space(self) -> np.ndarray:
        """
        Alias for Environemnt.state_space.

        Returns
        -------
        Space
            An object of type Space that represents possible states the agent may observe
        """
        return np.arange(self._grid_dims[0] * self._grid_dims[1])

    @property
    def action_space(self) -> np.ndarray:
        """
        The Space representing the range of possible actions.

        Returns
        -------
        Space
            An object of type Space that represents possible actions the agent may take
        """
        return spaces.Discrete(4)

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

    def device(self):
        """
        The torch device the environment lives on.
        """
        return self._device


