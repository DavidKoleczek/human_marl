from torch.optim import Adam
#from agents.vac import VAC
from all.approximation import VNetwork, FeatureNetwork
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.policies import SoftmaxPolicy
#from models import nature_features, nature_value_head, nature_policy_head
from models import simple_nature_features, simple_nature_value_head, simple_nature_policy_head

from torch.nn.functional import mse_loss
from all.agents._agent import Agent


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

    def act(self, state):
        self._train(state)
        self._features = self.features(state)
        self._distribution = self.policy(self._features)
        self._action = self._distribution.sample()
        return self._action

    def eval(self, state):
        return self.policy.eval(self.features.eval(state))

    def _train(self, state):
        if self._features:
            # forward pass
            values = self.v(self._features)

            # compute targets
            targets = state['reward'] + self.discount_factor * self.v.target(self.features.target(state))
            advantages = targets - values.detach()

            # compute losses
            value_loss = mse_loss(values, targets)
            policy_loss = -(advantages * self._distribution.log_prob(self._action)).mean()

            # backward pass
            self.v.reinforce(value_loss)
            self.policy.reinforce(policy_loss)
            self.features.reinforce()



def VAC_agent(
        # Common settings
        device="cuda",
        discount_factor=0.99,
        # Adam optimizer settings
        lr_v=5e-4,
        lr_pi=1e-4,
        eps=1.5e-4,
        # Other optimization settings
        clip_grad=0.5,
        value_loss_scaling=0.25,
        # Parallel actors
        n_envs=16,
        # Model construction
        feature_model_constructor=simple_nature_features,
        value_model_constructor=simple_nature_value_head,
        policy_model_constructor=simple_nature_policy_head
):
    """
    Vanilla Actor-Critic Atari preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr_v (float): Learning rate for value network.
        lr_pi (float): Learning rate for policy network and feature network.
        eps (float): Stability parameters for the Adam optimizer.
        clip_grad (float): The maximum magnitude of the gradient for any given parameter.
            Set to 0 to disable.
        value_loss_scaling (float): Coefficient for the value function loss.
        n_envs (int): Number of parallel environments.
        feature_model_constructor (function): The function used to construct the neural feature model.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """
    def _vac(envs, writer=DummyWriter()):
        value_model = value_model_constructor().to(device)
        #policy_model = policy_model_constructor(envs[0]).to(device)
        policy_model = policy_model_constructor(envs).to(device)
        feature_model = feature_model_constructor().to(device)

        value_optimizer = Adam(value_model.parameters(), lr=lr_v, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi, eps=eps)
        feature_optimizer = Adam(feature_model.parameters(), lr=lr_pi, eps=eps)

        v = VNetwork(
            value_model,
            value_optimizer,
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
            writer=writer,
        )
        policy = SoftmaxPolicy(
            policy_model,                                                                                
            policy_optimizer,
            clip_grad=clip_grad,
            writer=writer,
        )
        features = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad,
            writer=writer
        )

        return VAC(features, v, policy, discount_factor=discount_factor)
    #SingleEnvExperiment
    return _vac
    #ParallelEnvExperiment 
    #return _vac, n_envs
    
