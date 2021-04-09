import gym
import torch.nn as nn
from .architectures import Dueling, Linear0, CategoricalDueling, NoisyFactorizedLinear


def fc_q_discrete(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + env.action_space.n + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        Linear0(hidden2, 1),
    )


def fc_v_discrete(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        Linear0(hidden2, 1),
    )


def fc_soft_policy_discrete(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        Linear0(hidden2, env.action_space.n * 2),
    )
