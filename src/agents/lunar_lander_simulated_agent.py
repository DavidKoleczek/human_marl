import numpy as np
#laggy_pilot_policy = LaggyPilotPolicy(policy)
class LaggyPilotPolicy(object):
    def __init__(self, policy):
        self.last_laggy_pilot_act = None
        self.policy = policy

    def __call__(self, obs, lag_prob=0.8):
        action = self.policy.eval(obs)
        if self.last_laggy_pilot_act is None or np.random.random() >= lag_prob:
            self.last_laggy_pilot_act = action
        # else:
        #     print("laggy!")
        #     print("pilot_action:" + str(self.last_laggy_pilot_act) + " real action: " + str(action))
        return self.last_laggy_pilot_act

class NoisyPilotPolicy(object):
    def __init__(self, policy, noise_prob=0.15):
        self.policy = policy
        self.noise_prob = noise_prob

    def __call__(self, obs):
            action = self.policy.eval(obs)
            pilot_action = action
            if np.random.random() < self.noise_prob:
                action = (action + 3) % 6
            if np.random.random() < self.noise_prob:
                action = action//3*3 + (action + np.random.randint(1, 3)) % 3
            # if pilot_action != action:
            #     print("noisy!")
            # print("pilot_action:" + str(pilot_action) + " real action: " + str(action))
            return action

def noop_pilot_policy(obs):
    return 1

def sensor_pilot_policy(obs, thresh=0.1):
    obs = obs['observation']
    d = obs[8] - obs[0] # horizontal dist to helipad
    if d < -thresh:
        return 0
    elif d > thresh:
        return 2
    else:
        return 1