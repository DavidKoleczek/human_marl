from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from environments.hitl_sb_lunarlandercont import HITLSBLunarLanderCont
from agents.simulated.laggy import LaggyAgent
from agents.simulated.noisy import NoisyAgent
from agents.simulated.sensor import SensorAgent

import gym

model = SAC.load("sac_lunar.zip")
model = LaggyAgent(model, 0.8)

#eval_env = HITLSBLunarLanderCont('LunarLanderContinuous-v2', model)
eval_env = gym.make('LunarLanderContinuous-v2')
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)

print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
