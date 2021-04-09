import torch
import numpy as np

from easyrl.environments import State
from._replay_buffer import ReplayBuffer


class ExperienceReplayBuffer(ReplayBuffer):
    """Adapted from: https://github.com/Shmuma/ptan/blob/master/ptan/experience.py
    """

    def __init__(self, size, device=torch.device('cpu')):
        self.buffer = []
        self.capacity = int(size)
        self.pos = 0
        self.device = device

    def store(self, state, action, next_state):
        if state is not None and not state.done:
            self._add((state, action, next_state))

    def sample(self, batch_size):
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        minibatch = [self.buffer[key] for key in keys]
        return self._reshape(minibatch, torch.ones(batch_size, device=self.device))

    def update_priorities(self, td_errors):
        pass

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def _reshape(self, minibatch, weights):
        states = State.array([sample[0] for sample in minibatch])
        if torch.is_tensor(minibatch[0][1]):
            actions = torch.stack([sample[1] for sample in minibatch])
        else:
            actions = torch.tensor([sample[1] for sample in minibatch], device=self.device)
        next_states = State.array([sample[2] for sample in minibatch])
        return (states, actions, next_states.reward, next_states, weights)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)
