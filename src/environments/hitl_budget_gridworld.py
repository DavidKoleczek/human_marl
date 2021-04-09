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

    def __init__(self, human: Agent, intervention_penalty: float = 0.1, budget = 50):
        """Initializes a human in the loop wrapper around a GymEnvironment

        Args:
            human (Agent):  human should be a simulated agent that given the current state of the GymEnvironment will return some action
            device (str, optional): Defaults to torch.device('cpu').
        """
        super().__init__()
        self.human = human
        self.intervention_penalty = intervention_penalty
        self.interventions_made = 0
        self.unmodified_return = 0
        self.budget = budget
        self.remaining_budget = budget

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
            'observation': torch.cat((self._start_state, torch.tensor([0]), torch.tensor([self.budget])), 0),
            'reward': 0,
            'done': False
        })
        self._reward = 0
        self._action = None
        self._timestep = 0
        self._done = False
        self.interventions_made = 0
        self.unmodified_return = 0
        self.remaining_budget = self.budget
        return self._state

    def _trim_state(self, state: State) -> State:
        """ Removes the human action from the observation in the given State object

        Args:
            state (State): State object

        Returns:
            State: State object without the human action as part of the observation
        """
        modified_state = deepcopy(state)
        modified_state['observation'] = modified_state['observation'][:-2]
        return modified_state

    def _intervention_penalty(self) -> float:
        return self.intervention_penalty

    def step(self, action) -> State:
        """Overrides GymEnvironment's step method. Returns the state of the gym environment + the human's action.
        Modifies the reward depending on if the provided action constitutes an intervention

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """

        # get the human's action
        human_action = self.human.eval(self._trim_state(self.state))

        # if the agent did not the take noop action use its action
        # and treat it as an interventions
        if action != 8 and self.remaining_budget != 0:
            next_state = super().step(action)
            self.unmodified_return += float(next_state['reward'])
            next_state['reward'] -= self._intervention_penalty()
            self.interventions_made += 1
            self.remaining_budget -= 1
        # otherwise, use the human action
        else:
            next_state = super().step(human_action)
            self.unmodified_return += float(next_state['reward'])

        # concat state with the human's action, and remaining budget
        next_state['observation'] = torch.cat((next_state['observation'], torch.tensor([human_action]), torch.tensor([self.remaining_budget])), 0)
        return next_state

    @property
    def state_space(self) -> Space:
        """
        The Space representing the range of observable states.

        Returns
        -------
        Space
            An object of type Space that represents possible states the agent may observe
        """
        # add 2 to the size of the state space to accomodate the human's action, and budget
        shape = (self._grid_dims[0] * self._grid_dims[1]) + 2
        low = np.zeros(shape=shape, dtype=np.float32)
        high = np.ones(shape=shape, dtype=np.float32)
        # the upperbound on the human's action is the number of actions
        high[-2] = self.num_actions - 1
        # high should be budget interventions?
        high[-1] = self.budget
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

        return self.state_space

    @property
    def action_space(self) -> Space:
        """
        The Space representing the range of possible actions.

        Returns
        -------
        Space
            An object of type Space that represents possible actions the agent may take
        """
        # add 1 to the action space to accomdate the noop action
        return Discrete(8 + 1)

    @property
    def num_actions(self) -> int:
        """Get the number of actions for use in the Policy Network

        Returns:
            int: number of actions
        """
        # check if the gym has a discrete state space
        # add 1 to the action space to accomdate the noop action
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.action_space.n + 1

        raise ValueError('ES Agent currently only supports Discrete action spaces.')

    @property
    def get_interventions_made(self) -> int:
        """Get a count of interventions in this environment since the last reset.

        Returns:
            int: number of interventions
        """
        return self.interventions_made

    @property
    def get_unmodified_return(self) -> float:
        """Get the environment returns without considering the intervention penalty that was added

        Returns:
            float: environment returns
        """
        return self.unmodified_return
