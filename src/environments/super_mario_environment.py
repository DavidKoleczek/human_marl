import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp
from all.environments import GymEnvironment
import gym
from utils.lunar_lander_utils import disc_to_cont, onehot_encode, onehot_decode
from gym import spaces, wrappers
import types
import numpy as np
import torch
from all.core import State



class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], 0)
        return states.astype(np.float32)


def make_env(world, stage, action_type, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    #env = gym.wrappers.Monitor(env, "recording", force=True)
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    env = GymEnvironment(env, device="cuda")
    return env, env.observation_space.shape[0], len(actions)


class AddHumanAction(Wrapper):
    def __init__(self, env, pilot_policy):
        super(AddHumanAction, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.pilot_policy = pilot_policy

    def _convert(action):
        if torch.is_tensor(action):
            if isinstance(env.action_space, gym.spaces.Discrete):
                return action.item()
            if isinstance(env.action_space, gym.spaces.Box):
                return action.cpu().detach().numpy().reshape(-1)
            raise TypeError("Unknown action space type")
        return action

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        wrapper_state = State.from_gym(
            (state, reward, done, info),
            dtype=self.observation_space.dtype,
            device="cuda"
        )
        pilot_action = onehot_encode(self.pilot_policy(wrapper_state), self.env.action_space.n)
        info['pilot_action'] = torch.from_numpy(pilot_action).float().cuda()

        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        wrapper_state = State.from_gym(
            state,
            dtype=self.observation_space.dtype,
            device="cuda"
        )
        pilot_action = onehot_encode(self.pilot_policy(wrapper_state), self.env.action_space.n)
        reward = 0
        done = False
        info = {}
        info['pilot_action'] = torch.from_numpy(pilot_action).float().cuda()
        return state, reward, done, info


def make_co_env(world, stage, action_type, pilot_policy = None, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    #env = gym.wrappers.Monitor(env, "recording", force=True)
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    env = AddHumanAction(env, pilot_policy)
    env = GymEnvironment(env, device="cuda")
    return env, env.observation_space.shape[0], len(actions)



class AddHumanActionAndBudget(Wrapper):
    def __init__(self, env, pilot_policy, budget = 300):
        super(AddHumanActionAndBudget, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.pilot_policy = pilot_policy
        self.budget = budget
        self.remaining_budget = budget

    def _convert(action):
        if torch.is_tensor(action):
            if isinstance(env.action_space, gym.spaces.Discrete):
                return action.item()
            if isinstance(env.action_space, gym.spaces.Box):
                return action.cpu().detach().numpy().reshape(-1)
            raise TypeError("Unknown action space type")
        return action

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        wrapper_state = State.from_gym(
            (state, reward, done, info),
            dtype=self.observation_space.dtype,
            device="cuda"
        )
        pilot_action = onehot_encode(self.pilot_policy(wrapper_state), self.env.action_space.n)
        if self.budget == 0:
            info['pilot_action'] = torch.from_numpy(np.concatenate((pilot_action, [0]))).float().cuda()
        else:
            info['pilot_action'] = torch.from_numpy(np.concatenate((pilot_action, [self.remaining_budget / self.budget]))).float().cuda()
        return state, reward, done, info

    def reset(self):
        self.remaining_budget = self.budget
        state = self.env.reset()
        wrapper_state = State.from_gym(
            state,
            dtype=self.observation_space.dtype,
            device="cuda"
        )
        pilot_action = onehot_encode(self.pilot_policy(wrapper_state), self.env.action_space.n)
        reward = 0
        done = False
        info = {}
        if self.budget == 0:
            info['pilot_action'] = torch.from_numpy(np.concatenate((pilot_action, [0]))).float().cuda()
        else:
            info['pilot_action'] = torch.from_numpy(np.concatenate((pilot_action, [self.remaining_budget / self.budget]))).float().cuda()
        return state, reward, done, info

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



def make_co_budget_env(world, stage, action_type, pilot_policy = None, budget = 300, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    #env = gym.wrappers.Monitor(env, "recording", force=True)
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    env = AddHumanActionAndBudget(env, pilot_policy, budget)
    env = BudgetGymEnvironment(env, device="cuda")
    return env, env.observation_space.shape[0], len(actions)