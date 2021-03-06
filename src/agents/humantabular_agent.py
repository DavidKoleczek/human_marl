from typing import List

from all.agents import Agent
from all.core import State


class HumanTabularAgent(Agent):
    """Wrapper for a hardcoded policy
    """

    def __init__(self, tabular_policy: List[int]):
        self._tabular_policy = tabular_policy

    def act(self, state: State) -> int:
        """Gets the action given the current state
        """
        state = int(state['observation'].numpy()[0])
        return self._tabular_policy[state]

    def eval(self, state: State) -> int:
        """Gets the action given the current state
        """
        return self.act(state)
