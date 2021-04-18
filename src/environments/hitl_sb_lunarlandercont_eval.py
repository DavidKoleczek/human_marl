'''An environment for evaluating a trained agent with an actual human in the loop on LunarLanderContinuous

Follows the OpenAI Gym Environment Interface: https://github.com/openai/gym/blob/151ba406ebf07c5274c99542549aaacd6f70ba24/gym/core.py#L8
'''

import random
from copy import deepcopy

import numpy as np
import gym
from gym import spaces


class HITLSBLunarLanderContEval(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, env_name, hitl_agent, do_not_intervene=False):
        self.env = gym.make(env_name)
        self.hitl_agent = hitl_agent
        self.state = None
        self.do_not_intervene = do_not_intervene

        # logging variables
        self.interventions_made = 0
        self.non_noops_taken = 0
        self.raw_reward = 0
        self.modified_reward = 0
        self.timesteps = self.env._elapsed_steps

    @property
    def action_space(self):
        # one new action as a no-op
        new_shape = self.env.action_space.shape
        new_shape = (new_shape[0] + 1, )

        modified_action_space = spaces.Box(low=-1, high=1, shape=new_shape)

        # return modified_action_space
        return modified_action_space

    @property
    def observation_space(self):
        # add to the size of the state space to accomodate the human's action and intervention constraint
        new_shape = (self.env.observation_space.shape[0] + self.action_space.shape[0], )
        modified_obs_space = spaces.Box(low=np.NINF, high=np.Inf, shape=new_shape)
        return modified_obs_space

    @property
    def reward_range(self):
        return self.env.reward_range

    def _get_unmodified_state(self, state):
        """ Removes the human action from the observation in the given State object

        Args:
            state (State): State object

        Returns:
            State: State object without the human action as part of the observation
        """
        modified_state = deepcopy(state)
        modified_state = modified_state[:-3]
        return modified_state

    def _continuize_action(self, discrete_action):
        """A human's inputs are assumed to be discrete, so in order to use them
        in the trained SAC HITL agent, we make them "continuous".

        Args:
            discrete_action (np.ndarray):

        Returns:
            np.ndarray:
        """
        # no-op
        if discrete_action == 0:
            return np.array([0, 0, -1])
        # left engine
        elif discrete_action == 1:
            return np.array([0, -1, -1])
        # main engine
        elif discrete_action == 2:
            return np.array([1, 0, -1])
        # right engine
        elif discrete_action == 3:
            return np.array([0, 1, -1])
        # just in case
        else:
            return np.array([0, 0, -1])

    def step(self, action):
        self.timesteps = self.env._elapsed_steps  # logging

        # take the discrete human action and make it continuous
        human_action = self._continuize_action(action)

        # if hitl_agent is a list of agents, randomly choose one of them to be the hitl_action
        if isinstance(self.hitl_agent, list):
            which_agent = random.randrange(0, len(self.hitl_agent))
            hitl_agent = self.hitl_agent[which_agent]
            if self.env._elapsed_steps > 0:
                # for use with the HITL agent, we add the human's action to the underlying environment's state
                curr_state = np.concatenate((self.state[0][0: -self.action_space.shape[0]], human_action))
                # get the HITL agent's action
                hitl_action = hitl_agent.predict(curr_state)[0]
            else:
                # same idea as the if clause
                curr_state = np.concatenate((self.state[0: -self.action_space.shape[0]], human_action))
                hitl_action = hitl_agent.predict(curr_state)[0]
        else:
            # the difference between these if else clauses is that we need to handle the first timestep having a different format
            if self.env._elapsed_steps > 0:
                # for use with the HITL agent, we add the human's action to the underlying environment's state
                curr_state = np.concatenate((self.state[0][0: -self.action_space.shape[0]], human_action))
                # get the HITL agent's action
                hitl_action = self.hitl_agent.predict(curr_state)[0]
            else:
                # same idea as the if clause
                curr_state = np.concatenate((self.state[0: -self.action_space.shape[0]], human_action))
                hitl_action = self.hitl_agent.predict(curr_state)[0]

        # assume that the last element in the agent's action vector is what determines if it want's to take a no-op
        # if that element is less than 0, then it took a no-op, so we use the human's action
        did_intervene = True
        if hitl_action[-1] < 0 or self.do_not_intervene:
            did_intervene = False
            hitl_action = human_action

        state = self.env.step(hitl_action)
        self.raw_reward += state[1]  # logging

        # modifies the underlying environment's state in order to add the
        # current action to the observation and also add the intervention penalty
        temp = list(state)
        temp[0] = np.concatenate((state[0], hitl_action))
        state = tuple(temp)
        temp[3]['did_intervene'] = did_intervene

        self.state = state
        return self.state

    def reset(self):
        # reset logging variables
        self.interventions_made = 0
        self.non_noops_taken = 0
        self.raw_reward = 0
        self.modified_reward = 0
        self.timesteps = self.env._elapsed_steps

        # keep track of the state so we can provide it to the "human"
        self.state = self.env.reset()

        # the initial state will not take any action
        self.state = np.concatenate((self.state, np.array([0, 0, -1])))

        return self.state

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)
