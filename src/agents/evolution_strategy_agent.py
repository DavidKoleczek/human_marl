import gym
import torch
import torch.nn as nn

from all.agents import Agent


class SimplePolicyNetwork(nn.Module):
    def __init__(self, theta_size: int, num_actions: int):
        super().__init__()
        self.linear1 = nn.Linear(theta_size, 32)
        self.linear2 = nn.Linear(32, 32)
        self.actor_linear = nn.Linear(32, num_actions)

        # unsure why the reference implementation had this
        self.train()

    def forward(self, theta) -> torch.tensor:
        x = nn.SELU()(self.linear1(theta))
        x = nn.SELU()(self.linear2(x))
        outputs = self.actor_linear(x)

        # softmax action selection
        probs = torch.nn.functional.softmax(outputs, dim=-1)
        return torch.argmax(probs, dim=-1)

    def get_params(self):
        """ The network parameters that should be trained by ES (which is all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(), self.state_dict().values())]


class EvolutionStrategyAgent(Agent):
    """

    Attributes:
        :
    """

    def __init__(self, theta_size: int, num_actions: int):
        self.policy = SimplePolicyNetwork(theta_size, num_actions)

    def act(self, state):
        """
        """
        return 0

    def eval(self, state):
        """Chooses an action by a forward pass on the PolicyNetwork.
        """
        return self.policy.forward(state['observation'])
