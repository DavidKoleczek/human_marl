import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pickle
import random
import os
import math


from agents.super_mario_ddqn_agent import super_mario_DDQN_agent
from models.models import super_mario_nature_ddqn
from utils.super_mario_experiment import SuperMarioExperiment
from all.experiments.parallel_env_experiment import ParallelEnvExperiment
from environments.super_mario_environment import make_env

import argparse
import sys

def main(args):

    device = "cpu"
    if args.use_gpu:
        device = "cuda"

    max_timesteps = args.max_timesteps
    lr = args.lr
    discount_factor = args.discount_factor
    train = args.train
    load_pretrained_full_pilot = args.load_pretrained_full_pilot
    load_model_path = args.load_model_path
    

    env, num_states, num_actions = make_env(args.world, args.stage, args.action_type)

    agent = super_mario_DDQN_agent(
                    device = device, 
                    discount_factor = discount_factor, 
                    last_frame = max_timesteps,
                    lr = lr,
                    target_update_frequency = 1500, 
                    update_frequency = 4,
                    final_exploration = 0.02,
                    final_exploration_frame = 0.1 * max_timesteps,
                    prioritized_replay=True,
                    replay_start_size = 1000,
                    replay_buffer_size = 100000,
                    alpha=0.6,
                    beta=0.4,
                    model_constructor = super_mario_nature_ddqn)

    logdir='runs'
    quiet=False
    render=False
    write_loss=True

    path = "savedModels/super_mario/super_mario_pilot_model"
        
    make_experiment = SuperMarioExperiment
    experiment = make_experiment(
        agent,
        env,
        logdir=logdir,
        quiet=quiet,
        render=render,
        write_loss=write_loss,
        path = path
    )


    if load_pretrained_full_pilot:
        PATH = "savedModels/super_mario/" + load_model_path
        checkpoint = torch.load(PATH)
        experiment._agent.q.model.load_state_dict(checkpoint['q'])
        experiment._agent.policy.q.model.load_state_dict(checkpoint['q'])
    
    if train:
        experiment.train(frames=max_timesteps)
    else:
        experiment.show()
        experiment.test()


    from agents.lunar_lander_simulated_agent import sensor_pilot_policy, noop_pilot_policy, NoisyPilotPolicy, LaggyPilotPolicy

    noisy_pilot_policy = NoisyPilotPolicy(experiment._agent.policy, noise_prob = 0.25)
    laggy_pilot_policy = LaggyPilotPolicy(experiment._agent.policy)

    #experiment.show(policy = noop_pilot_policy)
    # experiment.show(policy = sensor_pilot_policy)
    # experiment.show(policy = laggy_pilot_policy)
    #experiment.show(policy = noisy_pilot_policy)

    #experiment.test(policy = noop_pilot_policy)
    # experiment.test(policy = sensor_pilot_policy)
    # experiment.test(policy = laggy_pilot_policy)
    experiment.test(policy = noisy_pilot_policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_timesteps", type=int, default=1e7)
    parser.add_argument('--discount_factor', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--load_pretrained_full_pilot', type=bool, default=False)
    parser.add_argument('--load_model_path', type=str, default=None)

    args = parser.parse_args()

    main(args)
