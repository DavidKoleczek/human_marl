from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    @abstractmethod
    def store(self, state, action, reward, next_state):
        '''Store the transition in the buffer'''

    @abstractmethod
    def sample(self, batch_size):
        '''Sample from the stored transitions'''

    @abstractmethod
    def update_priorities(self, indexes, td_errors):
        '''Update priorities based on the TD error'''
