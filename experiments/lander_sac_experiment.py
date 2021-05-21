"""Script to obtain results for the penalty method for Continuous Lunar Lander.
The RL algorithm used is Soft Actor Critic.
10 trials were run for the intervention penalties 0, 0.01, 0.1, 0.5, 1, 2, 5
"""

import os
import csv
from multiprocessing import Pool

from stable_baselines3 import SAC

from src.utils.lander_sac_optimal_agent import train_optimal_agent
from src.agents.simulated.sensor_sb import SensorAgent
from src.agents.simulated.laggy_sb import LaggyAgent
from src.agents.simulated.noisy_sb import NoisyAgent
from src.environments.sb.penalty_hitl import HITLLanderContinuous
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

    intervention_penalties = [0.01, 0.1, 0.5, 1, 3]
    # run 10 trials for each hyperparameter setting
    for _ in list(range(num_trials)):
        for penalty in intervention_penalties:
            env = HITLLanderContinuous('LunarLanderContinuous-v2', human_agent, intervention_penalty=penalty)
            model = SAC('MlpPolicy', env, verbose=1, tensorboard_log='runs')

            log_name = 'SAC={}_human={}'.format(penalty, human_agent.__class__.__name__ + str(human_parameter))
            output_path = './results/landersac_' + human_agent.__class__.__name__ + '.csv'

            model.learn(total_timesteps=300000, tb_log_name=log_name, callback=TensorboardCallback())
            model.save('./saved_models/lander_sac/lander_{}.zip'.format(log_name))

            # evaluate the trained agent
            eval_env = HITLLanderContinuous('LunarLanderContinuous-v2', human_agent,
                                            intervention_penalty=penalty, eval_mode=True)
            mean_reward, mean_int, mean_eplen, mean_int_rate = evaluate_policy_interventions(
                model, eval_env, n_eval_episodes=100, deterministic=True, return_episode_rewards=False)

            # save the results for each trial to a csv file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([penalty, mean_reward, mean_int, mean_eplen, mean_int_rate])

    return None


if __name__ == '__main__':
    human_experiment((SensorAgent, 0.1), 10)
