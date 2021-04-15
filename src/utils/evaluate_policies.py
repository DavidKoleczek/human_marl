import numpy as np
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


def average_policy_returns_hitl_sb(agent: Agent, env: Environment, num_episodes=1000, max_timesteps=500) -> float:
    """Computes the average returns, average unmodified returns, and total interventions
    of the given agent acting in a given human in the loop environment

    Args:
        agent (Agent): an Agent instance
        env (Environment): a HITL Environment instance
        num_episodes (int, optional): number of episodes to get an estimate
        max_timesteps (int, optional): maximum amount of timesteps per episode

    Returns:

    """
    total_returns = []
    total_interventions = []
    total_episode_length = []
    for _ in range(num_episodes):
        env.reset()
        episode_reward, interventions, episode_length = evaluate_policy_once_hitl_sb(agent, env, max_timesteps)
        total_returns.append(episode_reward)
        total_interventions.append(interventions)
        total_episode_length.append(episode_length)

    return total_returns, total_interventions, total_episode_length


def evaluate_policy_once_hitl_sb(agent: Agent, env: Environment, max_timesteps=1500) -> float:
    """Executes the given agent's policy for one episode in the given environment
    and returns the total reward, unmodified total reward, and total interventions

    Args:
        agent (Agent): an Agent instance
        env (Environment): a HITL Environment instance
        max_timesteps (int, optional): maximum amount of timesteps per episode

    Returns:

    """
    obs = env.reset()

    done, state = False, None
    episode_reward = 0.0
    episode_length = 0
    while not done and (episode_length < max_timesteps):
        action, state = agent.predict(obs, state=state, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        episode_length += 1

    total_interventions = env.interventions_made

    return episode_reward, total_interventions, episode_length


def compute_metrics_hitl_sb(agent: Agent, env: Environment, num_episodes=100, max_timesteps=1500):
    """Computes mean and standard error for: total return, total interventions, and inter
    """
    total_returns, total_interventions, total_episode_length = average_policy_returns_hitl_sb(agent, env, num_episodes, max_timesteps)

    total_returns = np.array(total_returns)
    total_interventions = np.array(total_interventions)
    total_episode_length = np.array(total_episode_length)
    intervention_rates = total_interventions / total_episode_length

    mean_return = np.mean(total_returns)
    mean_interventions = np.mean(total_interventions)
    mean_intervention_rate = np.mean(intervention_rates)

    stderr_return = np.std(total_returns) / np.sqrt(len(total_returns))
    stderr_interventions = np.std(total_interventions) / np.sqrt(len(total_interventions))
    stderr_intervention_rate = np.std(intervention_rates) / np.sqrt(len(total_interventions))

    return {
        'mean_return': mean_return,
        'mean_interventions': mean_interventions,
        'mean_intervention_rate': mean_intervention_rate,
        'stderr_return': stderr_return,
        'stderr_interventions': stderr_interventions,
        'stderr_intervention_rate': stderr_intervention_rate,
    }
