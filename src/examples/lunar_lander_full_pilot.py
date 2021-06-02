import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pickle
import random
import os
import math
import uuid
import time
from copy import copy
from collections import defaultdict, Counter
#import dill
import tempfile
import zipfile

from agents.ddqn_agent import DDQN_agent
from models.models import lunar_lander_nature_ddqn
from utils.lunar_lander_experiment import LundarLanderExperiment
from environments.lunar_lander_environment import make_env

#Every episode is at most 1000 steps. Use 500 episodes to train
max_ep_len = 1000
n_training_episodes = 500
max_timesteps = max_ep_len *  n_training_episodes

# If true, load the pretrained model. If false, train the model from the scratch.
load_pretrained_full_pilot = True

env = make_env(using_lander_reward_shaping=True)

agent = DDQN_agent(
                device = "cpu", 
                discount_factor = 0.99, 
                last_frame = max_timesteps,
                lr = 1e-3,
                target_update_frequency = 1500, 
                update_frequency = 1,
                final_exploration = 0.02,
                final_exploration_frame = 0.1 * max_timesteps,
                replay_start_size = 1000,
                replay_buffer_size = 50000,
                model_constructor = lunar_lander_nature_ddqn)

frames=max_timesteps
logdir='runs'
quiet=False
render=False
test_episodes=100
write_loss=True
    
make_experiment = LundarLanderExperiment
experiment = make_experiment(
    agent,
    env,
    logdir=logdir,
    quiet=quiet,
    render=render,
    write_loss=write_loss,
)

PATH = "savedModels/pilot_model.pkl"

if load_pretrained_full_pilot:
    checkpoint = torch.load(PATH)
    experiment._agent.q.model.load_state_dict(checkpoint['q'])
    experiment._agent.policy.q.model.load_state_dict(checkpoint['q'])
    experiment._agent.policy.epsilon = checkpoint['policy.epsilon']
else:
    experiment.train(frames=frames)
    model = experiment._agent
    state = {'q':model.q.model.state_dict(), 'policy.epsilon':model.policy.epsilon}
    torch.save(state, PATH)


experiment.show()
experiment.test()

from agents.lunar_lander_simulated_agent import sensor_pilot_policy, noop_pilot_policy, NoisyPilotPolicy, LaggyPilotPolicy

noisy_pilot_policy = NoisyPilotPolicy(experiment._agent.policy)
laggy_pilot_policy = LaggyPilotPolicy(experiment._agent.policy)

experiment.show(policy = noop_pilot_policy)
experiment.show(policy = sensor_pilot_policy)
experiment.show(policy = laggy_pilot_policy)
experiment.show(policy = noisy_pilot_policy)

print("noop")
experiment.test(policy = noop_pilot_policy)
print("sensor")
experiment.test(policy = sensor_pilot_policy)
print("laggy")
experiment.test(policy = laggy_pilot_policy)
print("noisy")
experiment.test(policy = noisy_pilot_policy)

