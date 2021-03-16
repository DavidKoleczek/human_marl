""" Human in the loop wrapper for GymEnvironments """
import gym
import torch

from all.environments.gym import GymEnvironment
from all.core import State


class HITLGymEnvironment(GymEnvironment):
    """ Human in the loop wrapper for GymEnvironments
    """

    def __init__(self, env, human, device=torch.device('cpu')):
        """Initializes a human in the loop wrapper around a GymEnvironment

        Args:
            env: either a string or an OpenAI gym environment
            human (TODO): TODO human should be a simulated agent that given the current state of the GymEnvironment will return some action
            device (str, optional): Defaults to torch.device('cpu').
        """
        super().__init__(env, device=torch.device('cpu'))
        self.human = human

    # TODO
    def step(self, action) -> State:
        """Overrides GymEnvironment's step method. Returns the state of the gym environment + the human's action.
        Modifies the reward depending on if the provided action constitutes an intervention

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """
        state = super().step(action)
        state['reward'] = torch.tensor(state['reward'])
        return state

    # TODO
    @property
    def state_space(self):
        return super().state_space

    # TODO
    @property
    def state(self):
        state = super().state
        return state

    @property
    def theta_size(self) -> int:
        """Gets the number of parameters of this env's state space which will determine
        the input of the PolicyNetwork.

        Returns:
            int: number of parameters of this env's state space
        """
        total_size = 0
        for s in self.observation_space.shape:
            total_size += s
        return total_size

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
