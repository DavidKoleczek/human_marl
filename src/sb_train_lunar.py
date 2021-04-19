'''Train an individual HITL agent on LunarLanderContinuous and then run a basic evaluation at the end.
One training run likely takes 1 to 1.5 hours on a mid-high end GPU.
'''

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from environments.hitl_sb_lunarcont_wbudget2 import HITLSBLunarContBudget
from agents.simulated.sensor import SensorAgent
from utils.tensorboard_callback import TensorboardCallback


human = SAC.load('savedModels/sac_lunar.zip')
human = SensorAgent(human, 0.1)

intervention_budget = 100
env = HITLSBLunarContBudget('LunarLanderContinuous-v2', human, intervention_budget=intervention_budget)
model = SAC('MlpPolicy', env, verbose=1, tensorboard_log='runs')

log_name = 'SAC_bud={}_human={}'.format(intervention_budget, 'Sensor0.1')
model.learn(total_timesteps=350000, tb_log_name=log_name, callback=TensorboardCallback())
model.save('savedModels/sac_lunar_{}.zip'.format(log_name))

# Evaluate the trained agent
eval_env = HITLSBLunarContBudget('LunarLanderContinuous-v2', human, intervention_budget=intervention_budget)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
print(f'mean_reward={mean_reward:.2f} +/- {std_reward}')
