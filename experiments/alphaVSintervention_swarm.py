import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np

from src.agents.ddqn_agent import DDQN_agent
from src.agents.co_ddqn_agent import co_DDQN_agent
from src.models.models import lunar_lander_nature_ddqn
from src.utils.lunar_lander_experiment import LundarLanderExperiment
from src.environments.lunar_lander_environment import make_co_env
from src.environments.lunar_lander_environment import make_env
from src.agents.lunar_lander_simulated_agent import sensor_pilot_policy, noop_pilot_policy, NoisyPilotPolicy, LaggyPilotPolicy

import argparse
import sys


def main(pilot_name="laggy_pilot", alpha=0, n_training_episodes=1000, train_pretrained_co_pilot=True):
    pilot_name = pilot_name
    alpha = alpha
    n_training_episodes = n_training_episodes
    train_pretrained_co_pilot = train_pretrained_co_pilot

    # dims for action and observation
    n_act_dim = 6
    n_obs_dim = 9

    # Every episode is at most 1000 steps. Use 500 episodes to train
    max_ep_len = 1000

    max_timesteps = max_ep_len * n_training_episodes

    env = make_env(using_lander_reward_shaping=True)

    agent = DDQN_agent(
        device="cpu",
        discount_factor=0.99,
        last_frame=max_timesteps,
        lr=1e-3,
        target_update_frequency=1500,
        update_frequency=1,
        final_exploration=0.02,
        final_exploration_frame=0.1 * max_timesteps,
        replay_start_size=1000,
        replay_buffer_size=50000,
        model_constructor=lunar_lander_nature_ddqn)

    frames = max_timesteps

    exp_pilot = LundarLanderExperiment(
        agent,
        env,
        logdir='runs',
        quiet=False,
        render=False,
        write_loss=False
    )

    PATH = os.path.abspath(os.path.join(os.getcwd(), "./", "saved_models/pilot_model.pkl"))

    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    exp_pilot._agent.q.model.load_state_dict(checkpoint['q'])
    exp_pilot._agent.policy.q.model.load_state_dict(checkpoint['q'])
    exp_pilot._agent.policy.epsilon = checkpoint['policy.epsilon']

    if pilot_name == "laggy_pilot":
        pilot_policy = LaggyPilotPolicy(exp_pilot._agent.policy)
    elif pilot_name == "noisy_pilot":
        pilot_policy = NoisyPilotPolicy(exp_pilot._agent.policy)
    elif pilot_name == "noop_pilot":
        pilot_policy = noop_pilot_policy
    elif pilot_name == "sensor_pilot":
        pilot_policy = sensor_pilot_policy

    episode_rewards = []
    episode_outcomes = []
    episode_interventions = []
    episode_steps = []
    for i in range(10):
        name = pilot_name + "_alpha_" + str(alpha)
        name = name + "_" + str(i)
        PATH = os.path.abspath(os.path.join(os.getcwd(), "./", "saved_models", name + ".pkl"))

        print("------------------------------------------------------")
        print(name)
        print("------------------------------------------------------")

        co_env = make_co_env(pilot_policy=pilot_policy, using_lander_reward_shaping=True)
        co_agent = co_DDQN_agent(
            device="cpu",
            discount_factor=0.99,
            last_frame=max_timesteps,
            lr=1e-3,
            target_update_frequency=1500,
            update_frequency=1,
            final_exploration=0.02,
            final_exploration_frame=0.1 * max_timesteps,
            replay_start_size=1000,
            replay_buffer_size=50000,
            model_constructor=lunar_lander_nature_ddqn,
            pilot_tol=alpha
        )

        frames = max_timesteps

        exp_co_pilot = LundarLanderExperiment(
            co_agent,
            co_env,
            logdir='runs',
            quiet=False,
            render=False,
            write_loss=False,
            name=name,
            path=PATH
        )

        if train_pretrained_co_pilot:
            exp_co_pilot.intervention_train(frames=frames)

        checkpoint = torch.load(PATH)
        exp_co_pilot._agent.q.model.load_state_dict(checkpoint['q'])
        exp_co_pilot._agent.policy.q.model.load_state_dict(checkpoint['q'])
        exp_co_pilot._agent.policy.epsilon = checkpoint['policy.epsilon']

        episode_reward, episode_outcome, episode_intervention, episode_step = exp_co_pilot.intervention_test()
        episode_rewards += episode_reward
        episode_outcomes += episode_outcome
        episode_interventions += episode_intervention
        episode_steps += episode_step

    episode_num = (i + 1) * 100 + 1

    mean_1000ep_reward = round(np.mean(episode_rewards[-episode_num:-1]), 1)
    std_1000ep_reward = round(np.std(episode_rewards[-episode_num:-1], ddof=1), 1)
    mean_1000ep_succ = round(np.mean([1 if x == 100 else 0 for x in episode_outcomes[-episode_num:-1]]), 2)
    mean_1000ep_crash = round(np.mean([1 if x == -100 else 0 for x in episode_outcomes[-episode_num:-1]]), 2)
    mean_1000ep_intervention = round(np.mean(episode_interventions[-episode_num:-1]), 1)
    std_1000ep_intervention = round(np.std(episode_interventions[-episode_num:-1], ddof=1), 1)
    mean_1000ep_step = round(np.mean(episode_steps[-episode_num:-1]), 1)
    std_1000ep_step = round(np.std(episode_steps[-episode_num:-1], ddof=1), 1)

    print("----------------------------------------------------------")
    print("mean 1000 episode reward", mean_1000ep_reward)
    print("std 1000 episode reward", std_1000ep_reward)
    print("mean 1000 episode intervention", mean_1000ep_intervention)
    print("std 1000 episode intervention", std_1000ep_intervention)
    print("mean 1000 episode steps", mean_1000ep_step)
    print("std 1000 episode steps", std_1000ep_step)
    print("mean 1000 episode succ", mean_1000ep_succ)
    print("mean 1000 episode crash", mean_1000ep_crash)
    print("----------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot_name', type=str, default="laggy_pilot")
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--n_training_episodes', type=int, default=1000)
    parser.add_argument('--train_pretrained_co_pilot', type=bool, default=True)
    args = vars(parser.parse_args())
    print(args)
    main(**args)
