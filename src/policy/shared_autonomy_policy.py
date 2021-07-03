import numpy as np
import torch
from all.optim import Schedulable
from src.utils.lunar_lander_utils import onehot_decode

class SharedAutonomyPolicy(Schedulable):

    def __init__(
                 self,
                 q,
                 num_action,
                 epsilon=0.,
                 pilot_tol=0
                ):
        self.q = q
        self.num_action = num_action
        self.epsilon = epsilon
        self.pilot_tol = pilot_tol

    def __call__(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_action)
        q_values = self.q.no_grad(state)
        pilot_action = onehot_decode(state.observation[-self.num_action:]) 
        return self.control_sharing(q_values, pilot_action)

    def no_grad(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_action)
        q_values = self.q.no_grad(state)
        pilot_action = onehot_decode(state.observation[-self.num_action:]) 
        return self.control_sharing(q_values, pilot_action)

    def eval(self, state):
        q_values = self.q.eval(state)
        pilot_action = onehot_decode(state.observation[-self.num_action:]) 
        return self.control_sharing(q_values, pilot_action)


    def control_sharing(self, q_values, pi_action):
        # alpha
        pilot_tol_ph = self.pilot_tol

        # copilot agent, opt for optimal
        q_values -= torch.min(q_values)
        opt_action = torch.argmax(q_values)
        opt_q_values = torch.max(q_values)

        # human action, pi for pilot
        pi_action = pi_action
        pi_act_q_values = q_values[pi_action]

        # if necessary, switch steering and keep main
        # try to use human's action for main engine, copilot's action for steering
        # evalute whether the new mixed action is suitable
        mixed_action = 3 * (pi_action // 3) + (opt_action % 3)
        # if >, still use human action, otherwise use mix action
        mixed_action = pi_action if pi_act_q_values >= (1 - pilot_tol_ph) * opt_q_values else mixed_action

        # if necessary, keep steering and switch main
        # try to use human's action for steering, copilot's action for main engine
        # evalute whether the new mixed action is suitable
        mixed_act_q_values = q_values[mixed_action]
        steer_mixed_action = 3 * (opt_action // 3) + (pi_action % 3)
        # if >, still use the preivous output, otherwise use steer_mix action
        mixed_action = mixed_action if mixed_act_q_values >= (1 - pilot_tol_ph) * opt_q_values else mixed_action

        # if necessary, switch steering and main
        # try to use copilot's action for main engine and steering
        # evalute whether the new mixed action is suitable
        mixed_act_q_values = q_values[mixed_action]
        action = mixed_action if mixed_act_q_values >= (1 - pilot_tol_ph) * opt_q_values else opt_action

        return action