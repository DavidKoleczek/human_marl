import os
import csv
import argparse

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from src.utils.lander_sac_optimal_agent import train_optimal_agent
from src.agents.simulated.sensor_sb import SensorAgent
from src.agents.simulated.laggy_sb import LaggyAgent
from src.agents.simulated.noisy_sb import NoisyAgent
from src.environments.sb.penaltyadapt_hitl import HITLLanderContinuousAdapt
from src.utils.eval_interventions import evaluate_policy_interventions
from src.utils.tensorboard_callback import TensorboardCallback


def human_experiment(human_params, num_trials):
    # we need an optimal agent to base our simulated agents
    if not os.path.exists('./saved_models/lander_sac_optimal.zip'):
        human = train_optimal_agent()
    else:
        human = SAC.load('./saved_models/lander_sac_optimal.zip')

    # create a simulated human agent
    human_parameter = human_params[1]
    human_agent = human_params[0](human, human_parameter)

    intervention_rates = [0.1, 0.25, 0.5, 0.75]
    # run 10 trials for each hyperparameter setting
    for _ in list(range(num_trials)):
        for rate in intervention_rates:
            env = HITLLanderContinuousAdapt('LunarLanderContinuous-v2', human,
                                            intervention_rate=rate, initial_intervention_penalty=1)
            model = SAC('MlpPolicy', env, verbose=1, tensorboard_log='runs')

            log_name = 'SACadapt={}_human={}'.format(rate, 'Sensor0.1')
            output_path = './results/landersacadapt_' + human_agent.__class__.__name__ + '.csv'

            model.learn(total_timesteps=350000, tb_log_name=log_name, callback=TensorboardCallback())
            model.save('./saved_models/lander_sac/lander_{}.zip'.format(log_name))

            # evaluate the trained agent
            eval_env = HITLLanderContinuousAdapt('LunarLanderContinuous-v2', human, eval_mode=True)
            mean_reward, mean_int, mean_eplen, mean_int_rate = evaluate_policy_interventions(
                model, eval_env, n_eval_episodes=100, deterministic=True, return_episode_rewards=False)

            # save the results for each trial to a csv file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([rate, mean_reward, mean_int, mean_eplen, mean_int_rate])

    return None


if __name__ == '__main__':
    human_experiment((SensorAgent, 0.1), 10)
