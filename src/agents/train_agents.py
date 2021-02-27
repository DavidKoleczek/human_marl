import numpy as np
from all.agents import Agent
from all.environments import Environment


def train_optimal_agent_tabularq(agent: Agent, env: Environment) -> Agent:
    # training the tabular Q-Learning agent
    for _ in range(1000):
        env.reset()
        state = env.state
        for _ in range(500):
            # choose action from state using a policy derived from the q function
            action = agent._epsilon_greedy_action_selection(state)

            # take that action and observe the reward and next state
            env.step(action)

            # train agent
            agent.act(state, action, env.state)

            if env.state['done']:
                break
            else:
                # set state to the next state
                state = env.state
    return agent
