from utils.lunar_lander_utils import disc_to_cont, onehot_encode, onehot_decode
from gym import spaces, wrappers
import gym
import types
from all.environments import GymEnvironment
import numpy as np
import torch
from all.core import State
from gym import Wrapper

# dims for action and observation
n_act_dim = 6
n_obs_dim = 9

def make_env(using_lander_reward_shaping=False):
    env = gym.make('LunarLanderContinuous-v2')
    #env = gym.wrappers.Monitor(env, "recording", force=True)
    env.action_space = spaces.Discrete(n_act_dim)
    
    #override the step function. Before run the originally step function, run disc_to_cont to convert 
    #the discrete action to continous action.
    env.unwrapped._step_orig = env.unwrapped.step
    def _step(self, action):
        obs, r, done, info = self._step_orig(disc_to_cont(action))
        return obs, r, done, info
    env.unwrapped.step = types.MethodType(_step, env.unwrapped)
    env.unwrapped.using_lander_reward_shaping = using_lander_reward_shaping

    env = GymEnvironment(env, device="cpu")
    return env

def make_co_env(pilot_policy, using_lander_reward_shaping=False):
    env = gym.make('LunarLanderContinuous-v2')
    #env = gym.wrappers.Monitor(env, "recording", force=True)
    env.action_space = spaces.Discrete(n_act_dim)
    env.unwrapped.pilot_policy = pilot_policy

    obs_box = env.observation_space
    env.observation_space = spaces.Box(np.concatenate((obs_box.low, np.zeros(n_act_dim))), 
                                         np.concatenate((obs_box.high, np.ones(n_act_dim))))

    #override the step function. Before run the originally step function, run disc_to_cont to convert 
    #the discrete action to continous action.
    env.unwrapped._step_orig = env.unwrapped.step
    env.unwrapped._reset_orig = env.unwrapped.reset

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
            device="cpu"
        )

        obs = state.observation.cpu().numpy()
        r = state.reward
        done = state.done
        info = {}

        pilot_action = onehot_encode(self.pilot_policy(state))
        obs = np.concatenate((obs, pilot_action))
        return obs, r, done, info

    def _reset(self):
        state = State.from_gym(self._reset_orig(), dtype=self.observation_space.dtype, device="cpu")
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

    env = GymEnvironment(env, device="cpu")
    return env


class AddHumanActionAndBudget(Wrapper):
    def __init__(self, env, pilot_policy, budget = 300):
        super(AddHumanActionAndBudget, self).__init__(env)
        self.pilot_policy = pilot_policy
        self.action_space = spaces.Discrete(n_act_dim)
        obs_box = self.env.observation_space
        self.observation_space = spaces.Box(np.concatenate((obs_box.low, np.zeros(n_act_dim + 1))), 
                                         np.concatenate((obs_box.high, np.ones(n_act_dim + 1))))
        self.budget = budget
        self.remaining_budget = budget

    def _convert(action):
        if torch.is_tensor(action):
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                return action.item()
            if isinstance(self.env.action_space, gym.spaces.Box):
                return action.cpu().detach().numpy().reshape(-1)
            raise TypeError("Unknown action space type")
        return action

    def step(self, action):
        state = State.from_gym(
            self.env.step(disc_to_cont(action)),
            dtype=self.observation_space.dtype,
            device="cpu"
        )

        obs = state.observation.cpu().numpy()
        reward = state.reward
        done = state.done
        info = {}

        pilot_action = onehot_encode(self.pilot_policy(state))
        if self.budget == 0:
            obs = np.concatenate((obs, pilot_action, [0]))
        else:
            obs = np.concatenate((obs, pilot_action, [self.remaining_budget / self.budget]))

        return obs, reward, done, info

    def reset(self):
        self.remaining_budget = self.budget
        state = State.from_gym(self.env.reset(), dtype=self.observation_space.dtype, device="cpu")
        obs = state.observation.cpu().numpy()
        r = state.reward
        done = state.done
        info = {}

        pilot_action = onehot_encode(self.pilot_policy(state))
        if self.budget == 0:
            obs = np.concatenate((obs, pilot_action, [0]))
        else:
            obs = np.concatenate((obs, pilot_action, [self.remaining_budget / self.budget]))
        return obs



from all.environments.abstract import Environment
gym.logger.set_level(40)

class BudgetGymEnvironment(Environment):
    def __init__(self, env, device=torch.device('cpu')):
        if isinstance(env, str):
            self._name = env
            env = gym.make(env)
        else:
            self._name = env.__class__.__name__

        self._env = env
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._info = None
        self._device = device

    @property
    def name(self):
        return self._name
    
    @property
    def remaining_budget(self):
        return self._env.remaining_budget
    
    @property
    def budget(self):
        return self._env.budget
    
    def buget_decrease(self):
        self._env.remaining_budget -= 1

    def reset(self):
        self._state = State.from_gym(self._env.reset(), dtype=self._env.observation_space.dtype, device=self._device)
        return self._state

    def step(self, action):
        self._state = State.from_gym(
            self._env.step(self._convert(action)),
            dtype=self._env.observation_space.dtype,
            device=self._device
        )
        return self._state

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    def duplicate(self, n):
        return [GymEnvironment(self._name, device=self.device) for _ in range(n)]

    @property
    def state_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def state(self):
        return self._state

    @property
    def env(self):
        return self._env

    @property
    def device(self):
        return self._device

    def _convert(self, action):
        if torch.is_tensor(action):
            if isinstance(self.action_space, gym.spaces.Discrete):
                return action.item()
            if isinstance(self.action_space, gym.spaces.Box):
                return action.cpu().detach().numpy().reshape(-1)
            raise TypeError("Unknown action space type")
        return action





def make_co_budget_env(pilot_policy, budget = 300, using_lander_reward_shaping=False):
    env = gym.make('LunarLanderContinuous-v2')
    #env = gym.wrappers.Monitor(env, "recording", force=True)
    env.unwrapped.using_lander_reward_shaping = using_lander_reward_shaping

    env = AddHumanActionAndBudget(env, pilot_policy, budget)
    env = BudgetGymEnvironment(env, device="cpu")
    return env