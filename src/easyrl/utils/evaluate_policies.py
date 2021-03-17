from easyrl.agents import Agent
from easyrl.environments import Environment


def average_policy_returns(agent: Agent, env: Environment, num_episodes=1000, max_timesteps=500) -> float:
    """Computes the average returns of the given agent acting in a given environment

    Args:
        agent (Agent): an Agent instance
        env (Environment): an Environment instance
        num_episodes (int, optional): number of episodes to get an estimate
        max_timesteps (int, optional): maximum amount of timesteps per episode

    Returns:
        float: average returns
    """
    total_returns = 0
    for _ in range(num_episodes):
        env.reset()
        total_returns += evaluate_policy_once(agent, env, max_timesteps)

    return total_returns / num_episodes


def evaluate_policy_once(agent: Agent, env: Environment, max_timesteps=500) -> float:
    """Executes the given agent's policy for one episode in the given environment

    Args:
        agent (Agent): an Agent instance
        env (Environment): an Environment instance
        max_timesteps (int, optional): maximum amount of timesteps per episode

    Returns:
        float: return for one episode
    """
    env.reset()
    total_reward = 0

    done = False
    timestep = 0
    while not done and (timestep < max_timesteps):
        action = agent.eval(env.state)
        env.step(action)

        total_reward += env.state['reward']
        done = env.state['done']
        timestep += 1

    return total_reward
