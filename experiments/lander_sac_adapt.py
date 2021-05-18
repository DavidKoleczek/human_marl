import os
import csv
from multiprocessing import Pool

from stable_baselines3 import SAC

from src.utils.lander_sac_optimal_agent import train_optimal_agent
from src.agents.simulated.sensor_sb import SensorAgent
from src.agents.simulated.laggy_sb import LaggyAgent
from src.agents.simulated.noisy_sb import NoisyAgent
from src.environments.sb.penaltyadapt_hitl import HITLLanderContinuousAdapt
from src.utils.eval_interventions import evaluate_policy_interventions

from src.utils.tensorboard_callback import TensorboardCallback


intervention_rate = 0.5

human = SAC.load('./saved_models/lander_sac_optimal.zip')
human = SensorAgent(human, 0.1)

env = HITLLanderContinuousAdapt('LunarLanderContinuous-v2', human,
                                intervention_rate=intervention_rate, initial_intervention_penalty=1)
model = SAC('MlpPolicy', env, verbose=1, tensorboard_log='runs')

log_name = 'SACadapt={}_human={}'.format(intervention_rate, 'Sensor0.1')
model.learn(total_timesteps=325000, tb_log_name=log_name, callback=TensorboardCallback())
model.save('./saved_models/lander_sac/lander_{}.zip'.format(log_name))


# Evaluate the trained agent
eval_env = HITLLanderContinuousAdapt('LunarLanderContinuous-v2', human)
mean_reward, mean_int, mean_eplen, mean_int_rate = evaluate_policy_interventions(
    model, eval_env, n_eval_episodes=100, deterministic=True, return_episode_rewards=False)

print(mean_reward, mean_int, mean_eplen, mean_int_rate)
