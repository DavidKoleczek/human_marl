''' Basic evaluation of HITL, SAC, Continuous LunarLander agents
'''

import gym
from stable_baselines3 import SAC

from agents.simulated.noisy import NoisyAgent
from environments.hitl_sb_lunarlandercont import HITLSBLunarLanderCont
from utils.evaluate_policies import compute_metrics_hitl_sb


human = SAC.load('savedModels/sac_lunar.zip')
human = NoisyAgent(human, 0.2)

hitl_agent = SAC.load('savedModels/sac_lunar_hitl_1p_sensor01.zip')
eval_env = HITLSBLunarLanderCont('LunarLanderContinuous-v2', human, intervention_penalty=0)


obs = eval_env.reset()
for i in range(1000):
    action, _states = hitl_agent.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    eval_env.render()
