from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import FixedTarget
from all.agents import DDQN
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer
from all.memory import ExperienceReplayBuffer
from all.nn import weighted_smooth_l1_loss
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from models.models import nature_ddqn
from models.super_mario_network import QNetwork


import torch
from all.nn import weighted_mse_loss
from all.agents._agent import Agent
from utils.lunar_lander_utils import onehot_decode
import numpy as np

class PADDQN(Agent):
    '''
    Double Deep Q-Network (DDQN).
    DDQN is an enchancment to DQN that uses a "double Q-style" update,
    wherein the online network is used to select target actions
    and the target network is used to evaluate these actions.
    https://arxiv.org/abs/1509.06461
    This agent also adds support for weighted replay buffers, such
    as priotized experience replay (PER).
    https://arxiv.org/abs/1511.05952

    Args:
        q (QNetwork): An Approximation of the Q function.
        policy (GreedyPolicy): A policy derived from the Q-function.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        discount_factor (float): Discount factor for future rewards.
        loss (function): The weighted loss function to use.
        minibatch_size (int): The number of experiences to sample in each training update.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    '''
    def __init__(self,
                 q,
                 policy,
                 replay_buffer,
                 penalty,
                 penalty_optimizer,
                 discount_factor=0.99,
                 loss=weighted_mse_loss,
                 minibatch_size=32,
                 replay_start_size=5000,
                 update_frequency=1,
                 budget_intervention_rate = 0,
                 act_num = 6,
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.loss = loss
        self.penalty = penalty
        self.penalty_optimizer = penalty_optimizer
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.budget_intervention_rate = budget_intervention_rate
        # private
        self._state = None
        self._action = None
        self._frames_seen = 0
        self._act_num = act_num

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
            (states, actions, rewards, next_states, weights) = self.replay_buffer.sample(self.minibatch_size)
            # forward pass
            values = self.q(states, actions)
            # compute targets
            next_actions = torch.argmax(self.q.no_grad(next_states), dim=1)
            targets = rewards + self.discount_factor * self.q.target(next_states, next_actions)

            pilot_actions = torch.nonzero(states["pilot_action"])[:,1]

            intervention_indicator = (pilot_actions != actions).int()
            penalty_targets = targets + self.penalty * (self.budget_intervention_rate - intervention_indicator)

            # compute loss
            value_loss = self.loss(values, penalty_targets, weights)
            # backward pass
            self.penalty.requires_grad = False
            self.q.reinforce(value_loss)

            self.penalty.requires_grad = True

            #lamda/penalty update
            penalty_loss = torch.mean(self.penalty * (self.budget_intervention_rate - intervention_indicator))
            penalty_loss.backward()
            self.penalty_optimizer.step()
            self.penalty_optimizer.zero_grad()

            if self.penalty[0] < 0:
                self.penalty.requires_grad = False
                self.penalty = (torch.ones(1) / 100000).cuda()
                self.penalty.requires_grad = True
            
            if self.penalty[0] > 100:
                self.penalty.requires_grad = False
                self.penalty = (torch.ones(1) * 100).cuda()
                self.penalty.requires_grad = True
                

            #update replay buffer priorities
            td_errors = penalty_targets - values
            self.replay_buffer.update_priorities(td_errors.abs())
            #print(self.penalty)




    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0

def super_mario_penalty_adapt_DDQN_agent(
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
        prioritized_replay=False,
        replay_start_size=80000,
        replay_buffer_size=1000000,
        # Explicit exploration
        initial_exploration=1.,
        final_exploration=0.01,
        final_exploration_frame=4000000,
        # Prioritized replay settings
        alpha=0.5,
        beta=0.5,
        # Model construction
        model_constructor=nature_ddqn,
        budget_intervention_rate = 0,
        act_num = 12
):
    """
    Double DQN with Prioritized Experience Replay (PER).

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
        initial_exploration (float): Initial probability of choosing a random action,
            decayed until final_exploration_frame.
        final_exploration (float): Final probability of choosing a random action.
        final_exploration_frame (int): The frame where the exploration decay stops.
        alpha (float): Amount of prioritization in the prioritized experience replay buffer.
            (0 = no prioritization, 1 = full prioritization)
        beta (float): The strength of the importance sampling correction for prioritized experience replay.
            (0 = no correction, 1 = full correction)
        model_constructor (function): The function used to construct the neural model.
    """
    def _ddqn(env, writer=DummyWriter()):
        action_repeat = 4
        last_timestep = last_frame / action_repeat
        last_update = (last_timestep - replay_start_size) / update_frequency
        final_exploration_step = final_exploration_frame / action_repeat

        model = model_constructor().to(device)
        value_optimizer = Adam(
            model.parameters(),
            lr=lr,
            eps=eps
        )

        penalty =torch.ones(1) / 1000
        penalty = penalty.cuda()
        penalty.requires_grad=True
        
        penalty_optimizer = Adam(
            [{'params': penalty}],
            lr=lr/10,
            eps=eps
        )

        q = QNetwork(
            model,
            value_optimizer,
            scheduler=CosineAnnealingLR(value_optimizer, last_update),
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
        
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(
                replay_buffer_size,
                alpha=alpha,
                beta=beta,
                device=device
            )
        else:
            replay_buffer = ExperienceReplayBuffer(
                replay_buffer_size,
                device=device
            )

        return PADDQN(q, policy, replay_buffer,
                 loss=weighted_smooth_l1_loss,
                 discount_factor=discount_factor,
                 minibatch_size=minibatch_size,
                 replay_start_size=replay_start_size,
                 update_frequency=update_frequency,
                 budget_intervention_rate = budget_intervention_rate,
                 act_num = act_num,
                 penalty = penalty,
                 penalty_optimizer = penalty_optimizer
                )
    return _ddqn
