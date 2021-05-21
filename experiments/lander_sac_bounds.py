'''Get the upper bound, or how well an optimal agent can perform at the environment.

Get the lower bound, or how well the human can do without any assistance
'''
import gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from src.agents.simulated.sensor_sb import SensorAgent
from src.environments.sb.penaltyadapt_hitl import HITLLanderContinuousAdapt


optimal = SAC.load('./saved_models/lander_sac_optimal.zip')
env = gym.make('LunarLanderContinuous-v2')
env.reset()

print(evaluate_policy(optimal, env, deterministic=True, n_eval_episodes=100))

env = gym.make('LunarLanderContinuous-v2')
env.reset()
human = SensorAgent(optimal, 0.1)
print(evaluate_policy(human, env, deterministic=True, n_eval_episodes=500))
