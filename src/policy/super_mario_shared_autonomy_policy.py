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
        pilot_action = onehot_decode(state["pilot_action"].cpu().numpy()) 
        return self.control_sharing(q_values, pilot_action)

    def no_grad(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_action)
        q_values = self.q.no_grad(state)
        
        pilot_action = onehot_decode(state["pilot_action"].cpu().numpy()) 
        return self.control_sharing(q_values, pilot_action)

    def eval(self, state):
        q_values = self.q.eval(state)
        pilot_action = onehot_decode(state["pilot_action"].cpu().numpy()) 
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

        noop = [0]
        right = [1,2,3,4]
        act = [5]
        left = [6,7,8,9]
        down = [10]
        up = [11]

        action_sequence = []
        if pi_action == 0:
            action_sequence += noop + up + down +  act + right + left
        elif pi_action >= 1 and pi_action <= 4:
            action_sequence += right + up + down +  act + noop + left
        elif pi_action == 5:
            action_sequence += act + up + down +  down + right + left + noop
        elif pi_action >= 6 and pi_action <= 9:
            action_sequence += left + up + down +  act + noop + right
        elif pi_action == 10:
            action_sequence += down + act + right + left + noop + up
        elif pi_action == 11:
            action_sequence += up +  act + right + left + noop + down

        for action in action_sequence:
            if q_values[action] >= (1 - pilot_tol_ph) * opt_q_values:
                return action


        return action