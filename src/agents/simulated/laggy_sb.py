'''Laggy "Pilot" implementation for LunarLanderContinuous-v2 and using the StableBaselines3 Agent API
'''

import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm


class LaggyAgent(BaseAlgorithm):
    def __init__(self, trained_agent, lag_prob=0.8):
        self.trained_agent = trained_agent
        self.lag_prob = lag_prob
        self.last_laggy_pilot_act = None

    def _setup_model(self) -> None:
        return super(LaggyAgent, self)._setup_model()

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=4,
        eval_env=None,
        eval_freq=-1,
        n_eval_episodes=5,
        tb_log_name="SAC",
        eval_log_path=None,
        reset_num_timesteps=True,
    ):
        """This function is just here to satisfy the BaseAlgorithm API
        """
        return super(LaggyAgent, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps)

    def _discretize_lunar(self, action):
        """Discretization of lunar lander continuous actions.
        See here for a description of the continuous action space https://github.com/openai/gym/blob/a5a6ae6bc0a5cfc0ff1ce9be723d59593c165022/gym/envs/box2d/lunar_lander.py#L104
        """
        # main engine off or on
        if action[0][0] < 0:
            action[0][0] = -1
        else:
            action[0][0] = 1

        # left, right, or off
        if action[0][1] <= -0.5:
            action[0][1] = -1
        elif action[0][1] >= 0.5:
            action[0][1] = 1
        else:
            action[0][1] = 0

        return action

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if self.last_laggy_pilot_act is None or np.random.random() >= self.lag_prob:
            action = self.trained_agent.predict(observation, state, mask, deterministic)
            self.last_laggy_pilot_act = self._discretize_lunar(action)
        return self.last_laggy_pilot_act
