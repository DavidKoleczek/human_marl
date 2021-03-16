from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import smooth_l1_loss
from all.approximation import QNetwork, FixedTarget
from all.agents import DQN
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from models.models import nature_dqn

import torch
from torch.nn.functional import mse_loss
from all.agents._agent import Agent


class DQN(Agent):
    '''
    Deep Q-Network (DQN).
    DQN was one of the original deep reinforcement learning algorithms.
    It extends the ideas behind Q-learning to work well with modern convolution networks.
    The core innovation is the use of a replay buffer, which allows the use of batch-style
    updates with decorrelated samples. It also uses a "target" network in order to
    improve the stability of updates.
    https://www.nature.com/articles/nature14236

    Args:
        q (QNetwork): An Approximation of the Q function.
        policy (GreedyPolicy): A policy derived from the Q-function.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        exploration (float): The probability of choosing a random action.
        loss (function): The weighted loss function to use.
        minibatch_size (int): The number of experiences to sample in each training update.
        n_actions (int): The number of available actions.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    '''
    def __init__(self,
                 q,
                 policy,
                 replay_buffer,
                 discount_factor=0.99,
                 loss=mse_loss,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1,
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.loss = loss
        # hyperparameters
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

    def act(self, state):
        self.replay_buffer.store(self._state, self._action, state)
        self._train()
        self._state = state
        self._action = self.policy.no_grad(state)
        return self._action

    def eval(self, state):
        return self.policy.eval(state)

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            values = self.q(states, actions)
            # compute targets
            targets = rewards + self.discount_factor * torch.max(self.q.target(next_states), dim=1)[0]
            # compute loss
            loss = self.loss(values, targets)
            # backward pass
            self.q.reinforce(loss)

    def _should_train(self):
        self._frames_seen += 1
        return (self._frames_seen > self.replay_start_size and
                self._frames_seen % self.update_frequency == 0)


def DQN_agent(
        # Common settings
        device="cuda",
        discount_factor=0.99,
        last_frame=40e6,
        # Adam optimizer settings
        lr=1e-4,
        eps=1.5e-4,
        # Training settings
        minibatch_size=32,
        update_frequency=4,
        target_update_frequency=1000,
        # Replay buffer settings
        replay_start_size=80000,
        replay_buffer_size=1000000,
        # Explicit exploration
        initial_exploration=1.,
        final_exploration=0.01,
        final_exploration_frame=4000000,
        # Model construction
        model_constructor=nature_dqn
):
    """
    DQN Atari preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (int): Initial probability of choosing a random action,
            decayed until final_exploration_frame.
        final_exploration (int): Final probability of choosing a random action.
        final_exploration_frame (int): The frame where the exploration decay stops.
        model_constructor (function): The function used to construct the neural model.
    """
    def _dqn(env, writer=DummyWriter()):
        action_repeat = 4
        last_timestep = last_frame / action_repeat
        last_update = (last_timestep - replay_start_size) / update_frequency
        final_exploration_step = final_exploration_frame / action_repeat

        model = model_constructor(env).to(device)

        optimizer = Adam(
            model.parameters(),
            lr=lr,
            eps=eps
        )

        q = QNetwork(
            model,
            optimizer,
            scheduler=CosineAnnealingLR(optimizer, last_update),
            target=FixedTarget(target_update_frequency),
            writer=writer
        )

        policy = GreedyPolicy(
            q,
            env.action_space.n,
            epsilon=LinearScheduler(
                initial_exploration,
                final_exploration,
                replay_start_size,
                final_exploration_step - replay_start_size,
                name="epsilon",
                writer=writer
            )
        )

        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )

        return DQN(
                q,
                policy,
                replay_buffer,
                discount_factor=discount_factor,
                loss=smooth_l1_loss,
                minibatch_size=minibatch_size,
                replay_start_size=replay_start_size,
                update_frequency=update_frequency,
                )
    return _dqn
