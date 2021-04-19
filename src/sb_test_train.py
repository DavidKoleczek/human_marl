'''Train an individual HITL agent on LunarLanderContinuous and then run a basic evaluation at the end.

One training run likely takes 1 to 1.5 hours on a mid-high end GPU.
'''

import numpy as np
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from eval_interventions import evaluate_policy_interventions


from environments.hitl_sb_lunarlandercont import HITLSBLunarLanderCont
from environments.hitl_sb_budget_lunarlandercont import HITLSBBudgetLunarLanderCont

from agents.simulated.sensor import SensorAgent
from utils.tensorboard_callback import TensorboardCallback
import sys

import argparse


def default_params():
    params = {}
    params['total_timesteps'] = 350000
    params['penalty'] = 1
    params['budget'] = 1000
    params['eval'] = 100
    
    return params


if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--total_timesteps', type = int, help = 'number of training timesteps')
    parser.add_argument('--penalty', type=float, help='penalty per intervention')
    parser.add_argument('--budget', type = int, help='intervention budget')
    parser.add_argument('--eval', type=int, help='number of evaluation episodes')
    

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    
    human = SAC.load('../savedModels/sac_lunar.zip')
    human = SensorAgent(human, 0.1)

    penalty = args.penalty
    budget = args.budget
    total_time = args.total_timesteps
    eval_ep = args.eval
    
    
    # env = HITLSBLunarLanderCont('LunarLanderContinuous-v2', human, intervention_penalty=penalty)
    env = HITLSBBudgetLunarLanderCont('LunarLanderContinuous-v2', human, intervention_penalty=penalty, budget = budget)

    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log='runs')

    log_name = 'SAC_ip={}_human={}'.format(penalty, 'Sensor0.1')
    model.learn(total_timesteps=total_time, tb_log_name=log_name, callback=TensorboardCallback())
    model.save('../savedModels/sac_lunar_hitl_{}p_{}b_sensor01.zip'.format(penalty, budget))

    # Evaluate the trained agent
    # eval_env = HITLSBLunarLanderCont('LunarLanderContinuous-v2', human)
    eval_env = HITLSBBudgetLunarLanderCont('LunarLanderContinuous-v2', human)

    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_ep, deterministic=True)
    rewards, lengths, interventions = evaluate_policy_interventions(model, eval_env, n_eval_episodes=eval_ep, deterministic=True, return_episode_rewards= True)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_int = np.mean(interventions)
    std_int = np.std(interventions)
    
    violations = [1 for i in interventions if i > budget]
    
    print(f'mean_reward={mean_reward:.2f} +/- {std_reward}')
    print(f'mean_intervention={mean_int:.2f} +/- {std_int}')
    print('Violations = {}/{}'.format(sum(violations),eval_ep))
    print(penalty, budget)