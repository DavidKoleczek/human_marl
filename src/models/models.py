import numpy as np
from all import nn
from torch import nn
import torch
import torch.nn.functional as F

def nature_dqn(env, frames=4):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n) 
    )

def nature_ddqn(env, frames=4):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dueling(
            nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear0(512, 1)
            ),
            nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear0(512, env.action_space.n)
            ),
        )
    )

def nature_features(frames=4):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
    )

def nature_value_head():
    return nn.Linear(512, 1)

def nature_policy_head(env):
    return nn.Linear0(512, env.action_space.n)

def nature_c51(env, frames=4, atoms=51):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n * atoms)
    )

def nature_rainbow(env, frames=4, hidden=512, atoms=51, sigma=0.5):
    return nn.Sequential(
        nn.Scale(1/255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.CategoricalDueling(
            nn.Sequential(
                nn.NoisyFactorizedLinear(3136, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            ),
            nn.Sequential(
                nn.NoisyFactorizedLinear(3136, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    env.action_space.n * atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            )
        )
    )

#designed for gridworld
def simple_nature_features(frames=4):
    return nn.Sequential(
        nn.Scale(1/4),
        nn.Linear(1, 16),
        nn.ReLU(),
    )

def simple_nature_value_head():
    return nn.Linear(16, 1)

def simple_nature_policy_head(env):
    return nn.Linear0(16, env.action_space.n)

def lunar_lander_nature_ddqn(env):
    return nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, env.action_space.n),
    )

def super_mario_nature_ddqn(env):
    return nn.Sequential(
        nn.Conv2d(env.observation_space.shape[0], 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 6 * 6, 512),
        nn.ReLU(),
        nn.Linear(512, env.action_space.n),
    )

class super_mario_co_ddqn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32 * 6 * 6, 512)
        self.linear2 = nn.Linear(512 + 12, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 12)



    def forward(self, pixel_state, action_state):
        pixel_state = F.relu(self.conv1(pixel_state))
        pixel_state = F.relu(self.conv2(pixel_state))
        pixel_state = F.relu(self.conv3(pixel_state))
        pixel_state = F.relu(self.conv4(pixel_state))
        pixel_state = self.flatten(pixel_state)
        pixel_state = F.relu(self.linear1(pixel_state))
        
        state = torch.cat((pixel_state, action_state), dim=1)

        state = F.relu(self.linear2(state))
        state = F.relu(self.linear3(state))
        state = self.linear4(state)
        return state


class super_mario_co_budget_ddqn(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32 * 6 * 6, 512)
        self.linear2 = nn.Linear(512 + 13, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 12)

    def forward(self, pixel_state, action_state):
        pixel_state = F.relu(self.conv1(pixel_state))
        pixel_state = F.relu(self.conv2(pixel_state))
        pixel_state = F.relu(self.conv3(pixel_state))
        pixel_state = F.relu(self.conv4(pixel_state))
        pixel_state = self.flatten(pixel_state)
        pixel_state = F.relu(self.linear1(pixel_state))
        
        state = torch.cat((pixel_state, action_state), dim=1)

        state = F.relu(self.linear2(state))
        state = F.relu(self.linear3(state))
        state = self.linear4(state)
        return state