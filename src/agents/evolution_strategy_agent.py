import gym
import torch
from all.agents import Agent


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # 8 is just hardcoding the network size for now, which is the same as the lunar lander state
        self.linear = torch.nn.Linear(8, num_actions)

    def forward(self, theta):
        outputs = self.linear(theta)

        # softmax action selection
        probs = torch.nn.functional.softmax(outputs, dim=-1)
        return torch.argmax(probs, dim=-1)


class EvolutionStrategyAgent(Agent):
    """

    Attributes:
        :
    """

    def __init__(self, action_space: gym.spaces.Space):
        # get the number of actions for the Policy Network
        if isinstance(action_space, gym.spaces.Discrete):
            self.num_actions = action_space.n
        else:
            raise ValueError('ES Agent currently only supports Discrete action spaces.')

        self.policy = PolicyNetwork(self.num_actions)

    def act(self, state):
        """
        """
        return 0

    def eval(self, state):
        """Chooses an action by a forward pass on the PolicyNetwork.
        """
        return self.policy.forward(state['observation'])
