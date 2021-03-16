from utils.lunar_lander_utils import disc_to_cont, onehot_encode, onehot_decode
from gym import spaces, wrappers
import gym
import types
from all.environments import GymEnvironment
import numpy as np
import torch
from all.core import State

# dims for action and observation
n_act_dim = 6
n_obs_dim = 9

def make_env(using_lander_reward_shaping=False):
    env = gym.make('LunarLanderContinuous-v2')
    env.action_space = spaces.Discrete(n_act_dim)

    #override the step function. Before run the originally step function, run disc_to_cont to convert 
    #the discrete action to continous action.
    env.unwrapped._step_orig = env.unwrapped.step
    def _step(self, action):
        obs, r, done, info = self._step_orig(disc_to_cont(action))
        return obs, r, done, info
    env.unwrapped.step = types.MethodType(_step, env.unwrapped)
    env.unwrapped.using_lander_reward_shaping = using_lander_reward_shaping

    env = GymEnvironment(env, device="cuda")
    return env

def make_co_env(pilot_policy, using_lander_reward_shaping=False):
    env = gym.make('LunarLanderContinuous-v2')
    env.action_space = spaces.Discrete(n_act_dim)
    env.unwrapped.pilot_policy = pilot_policy

    obs_box = env.observation_space
    env.observation_space = spaces.Box(np.concatenate((obs_box.low, np.zeros(n_act_dim))), 
                                         np.concatenate((obs_box.high, np.ones(n_act_dim))))

    #override the step function. Before run the originally step function, run disc_to_cont to convert 
    #the discrete action to continous action.
    env.unwrapped._step_orig = env.unwrapped.step
    env.unwrapped._reset_orig = env.unwrapped.reset

    # def _step(self, action):
    #     obs, r, done, info = self._step_orig(disc_to_cont(action))
    #     obs = np.concatenate(obs, onehot_encode(self.pilot_policy(obs)))
    #     return obs, r, done, info

    # def _reset(self):
    #     obs =  self._reset_orig()
    #     obs = np.concatenate(obs, onehot_encode(self.pilot_policy(obs)))
    #     return obs

    def _convert(action):
        if torch.is_tensor(action):
            if isinstance(env.action_space, gym.spaces.Discrete):
                return action.item()
            if isinstance(env.action_space, gym.spaces.Box):
                return action.cpu().detach().numpy().reshape(-1)
            raise TypeError("Unknown action space type")
        return action

    def _step(self, action):
        state = State.from_gym(
            self._step_orig(_convert(disc_to_cont(action))),
            dtype=self.observation_space.dtype,
            device="cuda"
        )

        obs = state.observation.cpu().numpy()
        r = state.reward
        done = state.done
        info = {}

        pilot_action = onehot_encode(self.pilot_policy(state))
        obs = np.concatenate((obs, pilot_action))
        return obs, r, done, info

    def _reset(self):
        state = State.from_gym(self._reset_orig(), dtype=self.observation_space.dtype, device="cuda")
        obs = state.observation.cpu().numpy()
        r = state.reward
        done = state.done
        info = {}

        pilot_action = onehot_encode(self.pilot_policy(state))
        obs = np.concatenate((obs, pilot_action))
        return obs

    env.unwrapped.step = types.MethodType(_step, env.unwrapped)
    env.unwrapped.reset = types.MethodType(_reset, env.unwrapped)
    env.unwrapped.using_lander_reward_shaping = using_lander_reward_shaping

    env = GymEnvironment(env, device="cuda")
    return env