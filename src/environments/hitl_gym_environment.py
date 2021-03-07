""" Human in the loop wrapper for GymEnvironments """
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
        return super().step(action)

    # TODO
    @property
    def state_space(self):
        return super().state_space

    # TODO
    @property
    def state(self):
        return super().state