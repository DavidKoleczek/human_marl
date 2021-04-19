''' A environment for training a human in the loop agent on LunarLanderContinuous

Follows the OpenAI Gym Environment Interface: https://github.com/openai/gym/blob/151ba406ebf07c5274c99542549aaacd6f70ba24/gym/core.py#L8
'''
from copy import deepcopy

import numpy as np
import gym
from gym import spaces


class HITLSBBudgetLunarLanderCont(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, env_name, human, intervention_penalty=0, budget = 1000):
        self.env = gym.make(env_name)
        self.human = human
        self.state = None
        self.intervention_penalty = intervention_penalty
        
        self.budget = 0
        self.remaining_budget = 0
        if budget != 0:
            self.budget = budget
            self.remaining_budget = 1.0
            
        # print(self.budget, self.remaining_budget)

        # logging variables
        self.interventions_made = 0
        self.non_noops_taken = 0
        self.raw_reward = 0
        self.modified_reward = 0
        self.timesteps = self.env._elapsed_steps

    @property
    def action_space(self):
        # one new action to be the no-op
        new_shape = self.env.action_space.shape
        new_shape = (new_shape[0] + 1, )

        modified_action_space = spaces.Box(low=-1, high=1, shape=new_shape)

        # return modified_action_space
        return modified_action_space

    @property
    def observation_space(self):
        # add to the size of the state space to accomodate the human's action and intervention constraint
        #Final +1 is to accomodate budget
        new_shape = (self.env.observation_space.shape[0] + self.action_space.shape[0] + 1, )
        low = np.full(new_shape, -np.inf)
        high = np.full(new_shape, np.inf)
        #Setting lowest allowed budget to 0
        low[-1] = 0
        modified_obs_space = spaces.Box(low=low, high=high, shape=new_shape)
        return modified_obs_space

    @property
    def reward_range(self):
        return self.env.reward_range

    def _get_unmodified_state(self, state):
        """ Removes the human action from the observation in the given State object

        Args:
            state (State): State object

        Returns:
            State: State object without the human action and budget as part of the observation
        """
        modified_state = deepcopy(state)
        modified_state = modified_state[:-4]
        return modified_state

    def step(self, action):
        self.timesteps = self.env._elapsed_steps  # logging

        # get the human's action.
        # handle if we are on the 0th timestep or not because the first state has a different form
        # it has just the observation; no "reward", "done", or "info" which is why this if else exists
        if self.env._elapsed_steps > 0:
            human_action = self.human.predict(self._get_unmodified_state(self.state[0]))
        else:
            human_action = self.human.predict(self._get_unmodified_state(self.state))

        # if the agent took the noop action (last value of action is less than or equal to 0), use the human action
        # otherwise penalize the agent, but take its action
        penalty = 0
        print(self.remaining_budget)
        if action[-1] <= 0 or self.remaining_budget == 0:
            action = human_action[0]  # indexing at 0 because the human action is wrapped in a tuple
            action = np.concatenate((action, np.array([-1])))  # -1 is basically adding to the state that an intervention was made (noop was made?)
        else:
            penalty = self.intervention_penalty
            self.non_noops_taken += 1  # logging
            self.interventions_made += 1  # logging
            self.remaining_budget -= 1/self.budget

        state = self.env.step(action)
        self.raw_reward += state[1]  # logging

        # modifies the underlying environment's state in order to add the
        # current action to the observation and also add the intervention penalty
        temp = list(state)
        temp[0] = np.concatenate((state[0], action, [self.remaining_budget]))
        temp[1] -= penalty
        state = tuple(temp)

        self.modified_reward += temp[1]  # logging

        self.state = state
        return self.state

    def reset(self):
        # reset logging variables
        self.interventions_made = 0
        self.non_noops_taken = 0
        self.raw_reward = 0
        self.modified_reward = 0
        self.timesteps = self.env._elapsed_steps
        if self.budget == 0:
            self.remaining_budget = 0
        else:
            self.remaining_budget = 1.0

        # keep track of the state so we can provide it to the "human"
        self.state = self.env.reset()
        human_action = self.human.predict(self.state)[0]

        # the initial state will use the first human actio and assume no intervention was made
        self.state = np.concatenate((self.state, human_action, np.array([-1]), [self.remaining_budget]))

        return self.state

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)
