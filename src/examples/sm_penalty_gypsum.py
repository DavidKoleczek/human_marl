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
from agents.super_mario_co_ddqn_agent import super_mario_co_DDQN_agent
from models.models import super_mario_nature_ddqn
from models.models import super_mario_co_ddqn
from utils.super_mario_experiment import SuperMarioExperiment
from all.experiments.parallel_env_experiment import ParallelEnvExperiment
from environments.super_mario_environment import make_env
from environments.super_mario_environment import make_co_env

import argparse
import sys

from agents.lunar_lander_simulated_agent import sensor_pilot_policy, noop_pilot_policy, NoisyPilotPolicy, LaggyPilotPolicy

def main(args):

    device = "cpu"
    if args.use_gpu:
        device = "cuda"
    #torch.manual_seed(12345)

    max_timesteps = args.max_timesteps
    lr = args.lr
    discount_factor = args.discount_factor
    train = args.train
    load_pretrained_full_pilot = args.load_pretrained_full_pilot
    load_model_path = args.load_model_path
    pilot_name = args.pilot_name
    intervention_punishment = args.intervention_punishment
    final_exploration = args.final_exploration
    final_exploration_frame_ratio =  args.final_exploration_frame_ratio
    num_models = args.num_models

    

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

        
    exp_pilot = SuperMarioExperiment(
        agent,
        env,
        logdir=logdir,
        quiet=quiet,
        render=render,
        write_loss=write_loss
    )

    #PATH = os.path.abspath(os.path.join(os.getcwd(), "../..", "savedModels/super_mario_pilot_model_reward_64.0.pkl"))

    PATH = "savedModels/super_mario_pilot_model_reward_64.0.pkl" 
    
    checkpoint = torch.load(PATH)
    exp_pilot._agent.q.model.load_state_dict(checkpoint['q'])
    exp_pilot._agent.policy.q.model.load_state_dict(checkpoint['q'])
    



    #pilot_policy = exp_pilot._agent.policy

    if pilot_name == "laggy_pilot":
        pilot_policy = LaggyPilotPolicy(exp_pilot._agent.policy)
    elif pilot_name == "noisy_pilot":
        pilot_policy = NoisyPilotPolicy(exp_pilot._agent.policy, noise_prob = 0.25)
    elif pilot_name == "noop_pilot":
        pilot_policy = noop_pilot_policy
    elif pilot_name == "sensor_pilot":
        pilot_policy = sensor_pilot_policy

    episode_rewards = [] 
    episode_outcomes = [] 
    episode_interventions = []
    episode_steps = []

    for i in range(num_models):
        name = pilot_name + "_penalty_" + str(intervention_punishment)
        name = name + "_" + str(i)
        #PATH = os.path.abspath(os.path.join(os.getcwd(), "../..", "savedModels/" + name)) 

        PATH = "savedModels/" + name

        print("------------------------------------------------------")
        print(name)
        print("------------------------------------------------------")

        co_env, num_states, num_actions = make_co_env(args.world, args.stage, args.action_type, pilot_policy)

        co_agent = super_mario_co_DDQN_agent(
                        device = device, 
                        discount_factor = discount_factor, 
                        last_frame = max_timesteps,
                        lr = lr,
                        target_update_frequency = 1500, 
                        update_frequency = 4,
                        final_exploration = final_exploration,
                        final_exploration_frame = final_exploration_frame_ratio * max_timesteps,
                        prioritized_replay=True,
                        replay_start_size = 1000,
                        replay_buffer_size = 100000,
                        alpha=0.6,
                        beta=0.4,
                        model_constructor = super_mario_co_ddqn)

        logdir='runs'
        quiet=False
        render=False
        write_loss=True
            
        exp_co_pilot = SuperMarioExperiment(
            co_agent,
            co_env,
            logdir=logdir,
            quiet=quiet,
            render=render,
            write_loss=write_loss,
            intervention_punishment = intervention_punishment,
            name = name,
            path = PATH
        )

        if load_pretrained_full_pilot:
            PATH = os.path.abspath(os.path.join(os.getcwd(), "../..", "savedModels/" + load_model_path))
            checkpoint = torch.load(PATH)
            exp_co_pilot._agent.q.model.load_state_dict(checkpoint['q'])
            exp_co_pilot._agent.policy.q.model.load_state_dict(checkpoint['q'])



        if train:
            exp_co_pilot.intervention_train(frames=max_timesteps)

        episode_reward, episode_outcome, episode_intervention, episode_step = exp_co_pilot.intervention_test()
        episode_rewards += episode_reward
        episode_outcomes += episode_outcome
        episode_interventions += episode_intervention
        episode_steps += episode_step


    episode_num = (i + 1) * 100 + 1

    
    mean_1000ep_reward = round(np.mean(episode_rewards[-episode_num:]), 1)
    std_1000ep_reward = round(np.std(episode_rewards[-episode_num:], ddof = 1), 1)
    mean_1000ep_succ = round(np.mean([1 if x==True else 0 for x in episode_outcomes[-episode_num:]]), 2)
    mean_1000ep_intervention = round(np.mean(episode_interventions[-episode_num:]), 1)
    std_1000ep_intervention = round(np.std(episode_interventions[-episode_num:], ddof = 1), 1)
    mean_1000ep_step = round(np.mean(episode_steps[-episode_num:]), 1)
    std_1000ep_step = round(np.std(episode_steps[-episode_num:], ddof = 1), 1)

    print("----------------------------------------------------------")
    print("mean 1000 episode reward", mean_1000ep_reward)
    print("std 1000 episode reward", std_1000ep_reward)
    print("mean 1000 episode intervention", mean_1000ep_intervention)
    print("std 1000 episode intervention", std_1000ep_intervention)
    print("mean 1000 episode steps", mean_1000ep_step)
    print("std 1000 episode steps", std_1000ep_step)
    print("mean 1000 episode succ", mean_1000ep_succ)
    print("----------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--max_timesteps", type=int, default=2e6)
    parser.add_argument('--discount_factor', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--load_pretrained_full_pilot', type=bool, default=None)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--pilot_name', type=str, default="noisy_pilot")
    parser.add_argument('--intervention_punishment', type=float, default=50)
    parser.add_argument('--final_exploration', type=float, default=0.05)
    parser.add_argument('--final_exploration_frame_ratio', type=float, default=0.1)
    parser.add_argument('--num_models', type=int, default=1)



    args = parser.parse_args()
    print(args)


    main(args)
