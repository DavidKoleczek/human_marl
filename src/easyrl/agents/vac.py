from torch.nn.functional import mse_loss
from torch.optim import Adam

from easyrl.approximation import VNetwork, FeatureNetwork
from easyrl.approximation.layers import fc_relu_features, fc_value_head, fc_policy_head
from easyrl.policies import SoftmaxPolicy
from easyrl.logging import DummyWriter
from .agent import Agent


class VAC(Agent):
    '''
    Vanilla Actor-Critic (VAC).
    VAC is an implementation of the actor-critic alogorithm found in the Sutton and Barto (2018) textbook.
    This implementation tweaks the algorithm slightly by using a shared feature layer.
    It is also compatible with the use of parallel environments.
    https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf

    Args:
        features (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (StochasticPolicy): Policy head which outputs an action distribution.
        discount_factor (float): Discount factor for future rewards.
        n_envs (int): Number of parallel actors/environments
        n_steps (int): Number of timesteps per rollout. Updates are performed once per rollout.
        writer (Writer): Used for logging.
    '''

    def __init__(self, features, v, policy, discount_factor=1):
        self.features = features
        self.v = v
        self.policy = policy
        self.discount_factor = discount_factor
        self._features = None
        self._distribution = None
        self._action = None

    @property
    def name(self) -> str:
        """
        The name of the agent.
        """
        return 'VanillaActorCritic'

    def act(self, state):
        self._train(state, state.reward)
        self._features = self.features(state)
        self._distribution = self.policy(self._features)
        self._action = self._distribution.sample()
        return self._action

    def eval(self, state):
        return self.policy.eval(self.features.eval(state))

    def _train(self, state, reward):
        if self._features:
            # forward pass
            values = self.v(self._features)

            # compute targets
            targets = reward + self.discount_factor * self.v.target(self.features.target(state))
            advantages = targets - values.detach()

            # compute losses
            value_loss = mse_loss(values, targets)
            policy_loss = -(advantages * self._distribution.log_prob(self._action)).mean()

            # backward pass
            self.v.reinforce(value_loss)
            self.policy.reinforce(policy_loss)
            self.features.reinforce()


def vac_preset(env,
               device="cpu",
               discount_factor=0.99,
               # Adam optimizer settings
               lr_v=5e-3,
               lr_pi=1e-3,
               eps=1e-5,
               # Model construction
               feature_model_constructor=fc_relu_features,
               value_model_constructor=fc_value_head,
               policy_model_constructor=fc_policy_head,
               writer=DummyWriter()):
    """
    Vanilla Actor-Critic classic control preset.
    Args:
        env (Environment): an Environment instance, used for initialization
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr_v (float): Learning rate for value network.
        lr_pi (float): Learning rate for policy network and feature network.
        eps (float): Stability parameters for the Adam optimizer.
        feature_model_constructor (function): The function used to construct the neural feature model.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """

    value_model = value_model_constructor().to(device)
    policy_model = policy_model_constructor(env).to(device)
    feature_model = feature_model_constructor(env).to(device)

    value_optimizer = Adam(value_model.parameters(), lr=lr_v, eps=eps)
    policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi, eps=eps)
    feature_optimizer = Adam(feature_model.parameters(), lr=lr_pi, eps=eps)

    v = VNetwork(value_model, value_optimizer, writer=writer)
    policy = SoftmaxPolicy(policy_model, policy_optimizer, writer=writer)
    features = FeatureNetwork(feature_model, feature_optimizer)

    return VAC(features, v, policy, discount_factor=discount_factor)
