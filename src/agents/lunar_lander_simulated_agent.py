import numpy as np

from all.agents import Agent


class LaggyPilotPolicy(object):
    def __init__(self, policy):
        self.last_laggy_pilot_act = None
        self.policy = policy

    def __call__(self, obs, lag_prob=0.8):
        if self.last_laggy_pilot_act is None or np.random.random() >= lag_prob:
            action = self.policy.eval(obs)
            self.last_laggy_pilot_act = action
        return self.last_laggy_pilot_act


class NoisyPilotPolicy(object):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, obs, noise_prob=0.15):
        action = self.policy.eval(obs)
        if np.random.random() < noise_prob:
            action = (action + 3) % 6
        if np.random.random() < noise_prob:
            action = action // 3 * 3 + (action + np.random.randint(1, 3)) % 3
        return action


def noop_pilot_policy(obs):
    return 1


def sensor_pilot_policy(obs, thresh=0.1):
    obs = obs['observation']
    d = obs[8] - obs[0]  # horizontal dist to helipad
    if d < -thresh:
        return 0
    elif d > thresh:
        return 2
    else:
        return 1


class LaggyPilotPolicyAgent(object):
    def __init__(self, agent: Agent, lag_prob=0.8):
        self.agent = agent
        self.lag_prob = lag_prob
        self.last_laggy_pilot_act = None

    def eval(self, state):
        if self.last_laggy_pilot_act is None or np.random.random() >= self.lag_prob:
            action = self.agent.eval(state)
            self.last_laggy_pilot_act = action
        return self.last_laggy_pilot_act
