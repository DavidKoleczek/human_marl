''' Basic evaluation of HITL, SAC, Continuous LunarLander agents
'''

import gym
from stable_baselines3 import SAC
import pandas as pd
import numpy as np

from agents.simulated.noisy import NoisyAgent
from environments.hitl_sb_lunarlandercont import HITLSBLunarLanderCont
from utils.evaluate_policies import compute_metrics_hitl_sb


human = SAC.load('savedModels/sac_lunar.zip')
human = NoisyAgent(human, 0.2)

hitl_agent = SAC.load('savedModels/sac_lunar_hitl_1p_sensor01.zip')
eval_env = HITLSBLunarLanderCont('LunarLanderContinuous-v2', human, intervention_penalty=0)


dataset = []
obs = eval_env.reset()
episode_counter = 0
for i in range(50000):
    action, _states = hitl_agent.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    dataset.append(np.append(obs, [episode_counter]))
    eval_env.render()
    if dones:
        obs = eval_env.reset()
        episode_counter += 1
    # track progress
    if i % 1000 == 0:
        print(i)


dataset = pd.DataFrame(dataset, columns=['Pos Horizontal', 'Pos Vertical', 'Speed Horizontal', 'Speed Vertical', 'Angle',
                                         'Angular Speed', 'Leg 1 Contact', 'Leg 2 Contact', 'Main Engine Action', 'Left-Right Action', 'Was Intervention', 'Episode Number'])
dataset.to_csv('plots/data_sac_lunar_hitl_1p_sensor01.csv', index=False)
