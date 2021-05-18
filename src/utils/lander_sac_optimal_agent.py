""" Train an optimal agent for Lunar Lander using Soft Actor Critic.
This is used as a base for the simulated humans.
"""
import gym
from stable_baselines3 import SAC


def train_optimal_agent():
    env = gym.make('LunarLanderContinuous-v2')

    model = SAC('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=550000)
    model.save('saved_models/lander_sac_optimal.zip')
    return model
