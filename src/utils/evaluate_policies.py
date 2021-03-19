from all.agents import Agent
from all.environments import Environment


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


def average_policy_returns_hitl(agent: Agent, env: Environment, num_episodes=1000, max_timesteps=500) -> float:
    """Computes the average returns, average unmodified returns, and total interventions
    of the given agent acting in a given human in the loop environment

    Args:
        agent (Agent): an Agent instance
        env (Environment): a HITL Environment instance
        num_episodes (int, optional): number of episodes to get an estimate
        max_timesteps (int, optional): maximum amount of timesteps per episode

    Returns:
        float: average returns
    """
    total_returns = 0
    total_interventions = 0
    total_unmodified_reward = 0
    for _ in range(num_episodes):
        env.reset()
        returns, unmodified_total_reward, interventions = evaluate_policy_once_hitl(agent, env, max_timesteps)
        total_returns += returns
        total_unmodified_reward += unmodified_total_reward
        total_interventions += interventions

    return total_returns / num_episodes, total_unmodified_reward / num_episodes, total_interventions / num_episodes


def evaluate_policy_once_hitl(agent: Agent, env: Environment, max_timesteps=500) -> float:
    """Executes the given agent's policy for one episode in the given environment
    and returns the total reward, unmodified total reward, and total interventions

    Args:
        agent (Agent): an Agent instance
        env (Environment): a HITL Environment instance
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

    total_inverventions = env.get_interventions_made
    unmodified_total_reward = env.get_unmodified_return
    return total_reward, unmodified_total_reward, total_inverventions
