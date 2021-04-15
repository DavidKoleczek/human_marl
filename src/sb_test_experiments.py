'''Runs a series of experiments over various different parameters, see main function for more details.
'''

import csv
from multiprocessing import Pool

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from environments.hitl_sb_lunarlandercont import HITLSBLunarLanderCont
from utils.tensorboard_callback import TensorboardCallback
from agents.simulated.laggy import LaggyAgent
from agents.simulated.noisy import NoisyAgent
from agents.simulated.sensor import SensorAgent


def experiment(params):
    human = SAC.load("sac_lunar.zip")
    human = params[1](human, params[2])
    params = list(params)
    params[1] = params[1].__name__
    params = tuple(params)

    env = HITLSBLunarLanderCont('LunarLanderContinuous-v2', human, intervention_penalty=params[0])

    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="runs")

    log_name = 'SAC_ip={}_human={}'.format(params[0], params[1])
    model.learn(total_timesteps=300000, tb_log_name=log_name, callback=TensorboardCallback())

    # Evaluate the trained agent
    eval_env = HITLSBLunarLanderCont('LunarLanderContinuous-v2', human)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    return mean_reward, std_reward, params


if __name__ == '__main__':
    # Run experiments in parallel
    # Various values of intervention penalty: 0, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10
    # Three different pilots: Laggy, Noisy, Sensor and their corresponding hyperparameter
    penalties = [0, 0.01, 0.1, 0.5, 1, 5, 10]
    agents = [(LaggyAgent, 0.8), (NoisyAgent, 0.15), (SensorAgent, 0.1)]

    parameters = []
    for i in penalties:
        for j in agents:
            parameters.append((i, j[0], j[1]))

    with Pool(processes=4) as p:
        res = p.map(experiment, parameters)

    with open('results3.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['mean_reward', 'std_reward', 'penalty', 'human', 'human_parameter'])
        for r in res:
            csvwriter.writerow((r[0], r[1], r[2][0], r[2][1], r[2][2]))
