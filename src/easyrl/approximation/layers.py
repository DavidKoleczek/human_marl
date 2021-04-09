import gym
import torch.nn as nn
from .architectures import Dueling, Linear0, CategoricalDueling, NoisyFactorizedLinear


def fc_relu_q(env, hidden=64):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], hidden),
        nn.ReLU(),
        nn.Linear(hidden, env.action_space.n),
    )


def dueling_fc_relu_q(env):
    return nn.Sequential(
        nn.Flatten(),
        Dueling(
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256), nn.ReLU(), nn.Linear(256, 1)
            ),
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, env.action_space.n),
            ),
        ),
    )


def fc_relu_features(env, hidden=64):
    if isinstance(env.state_space, gym.spaces.Box):
        return nn.Sequential(nn.Flatten(), nn.Linear(env.state_space.shape[0], hidden), nn.ReLU())
    else:
        raise ValueError('space_space of type {} is not supported in easyrl.approximation.layers.fc_relu_features'.format(type(env.state_space)))


def fc_relu_features_discrete(env, hidden=64):
    return nn.Sequential(nn.Flatten(), nn.Linear(env.state_space.shape[0], hidden), nn.ReLU())


def fc_value_head(hidden=64):
    return Linear0(hidden, 1)


def fc_policy_head(env, hidden=64):
    return Linear0(hidden, env.action_space.n)


def fc_relu_dist_q(env, hidden=64, atoms=51):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], hidden),
        nn.ReLU(),
        Linear0(hidden, env.action_space.n * atoms),
    )


def fc_relu_rainbow(env, hidden=64, atoms=51, sigma=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], hidden),
        nn.ReLU(),
        CategoricalDueling(
            NoisyFactorizedLinear(hidden, atoms, sigma_init=sigma),
            NoisyFactorizedLinear(
                hidden, env.action_space.n * atoms, init_scale=0.0, sigma_init=sigma
            ),
        ),
    )


def fc_q(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        Linear0(hidden2, 1),
    )


def fc_v(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        Linear0(hidden2, 1),
    )


def fc_soft_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        Linear0(hidden2, env.action_space.shape[0] * 2),
    )


def fc_actor_critic(env, hidden1=400, hidden2=300):
    features = nn.Sequential(
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
    )

    v = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1)
    )

    policy = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2)
    )

    return features, v, policy
