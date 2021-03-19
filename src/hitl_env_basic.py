'''Trying traditional RL algorithms on our Human in the Loop setup for modified 687 Gridworld.
'''
import numpy as np
import pandas as pd

from easyrl.environments import GridworldEnvironment
from easyrl.agents import vac_preset
from easyrl.train.single_env_experiment import SingleEnvExperiment
from easyrl.utils import average_policy_returns
from utils.evaluate_policies import average_policy_returns_hitl
from utils.misc import std_error
from agents.lunar_lander_simulated_agent import LaggyPilotPolicyAgent
from environments.hitl_gridworld import HITLGridworldEnvironment


env = GridworldEnvironment()

# we need a fake human agent
# first train an optimal agent using Vanilla Actor-Critic (VAC)
# keep trying VAC training until we get a good agent, sometimes it does not converge
human_performance = 0  # how good our fake agent does on the regular environment
while human_performance < 3:
    fake_human_agent = vac_preset(env)
    trainer = SingleEnvExperiment(fake_human_agent, env, quiet=True)
    trainer.train(episodes=600)
    human_performance = average_policy_returns(fake_human_agent, env, num_episodes=100)
    print('Fake human policy, no lag, average performance (reward): ', human_performance)

# wrap the optimal agent with a LaggyPilotPolicy agent, which lag_prob of the time selects the previous action
fake_human_agent = LaggyPilotPolicyAgent(fake_human_agent, lag_prob=0.75)
# evaluate how good this fake human does
human_performance = average_policy_returns(fake_human_agent, env, num_episodes=100)
print('Fake human policy average performance (reward): ', human_performance)


# we evaluate the performance of our human in the loop setup over different reward penalities
# each penalty is executed over x trials to minimize variance in our results
TRIALS = 20
penalties = [0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
avg_returns = []
avg_returns_stderr = []
avg_unmodified_returns = []
avg_unmodified_returns_sterr = []
avg_interventions = []
avg_interventions_sterr = []
for penalty in penalties:
    return_history, unmodified_returns_history, interventions_history = [], [], []
    for t in range(TRIALS):
        print('Penalty {}: Trial {}'.format(penalty, t))

        hitl_env = HITLGridworldEnvironment(fake_human_agent, intervention_penalty=penalty)

        # this will be our optimally trained agent for the new environment
        agent = vac_preset(hitl_env)

        # train the agent
        experiment = SingleEnvExperiment(agent, hitl_env, quiet=True)
        experiment.train(episodes=600)

        # get the returns achieved by the environment (with the intervention penalty),
        # the returns without the intervention penalty,
        # and the discrete amount of interventions taken by the agent
        returns, unmodified_returns, interventions = average_policy_returns_hitl(agent, hitl_env, num_episodes=100)

        # keep track of the metrics for this trial
        return_history.append(returns)
        unmodified_returns_history.append(unmodified_returns)
        interventions_history.append(interventions)

    # for each penalty over all trials, we store the average value and standard error
    # this is done for three metrics: average returns,
    # average unmodified returns (does not consider the intervention penalty),
    # and the discrete count of interventions made
    avg_returns.append(np.array(return_history).mean())
    avg_returns_stderr.append(std_error(np.array(return_history)))

    avg_unmodified_returns.append(np.array(unmodified_returns_history).mean())
    avg_unmodified_returns_sterr.append(std_error(np.array(unmodified_returns_history)))

    avg_interventions.append(np.array(interventions_history).mean())
    avg_interventions_sterr.append(std_error(np.array(interventions_history)))


# save data for plotting
data_to_plot = pd.DataFrame({
    'Penalty': penalties,
    'Avg Returns': avg_returns,
    'Avg Returns Std Error': avg_returns_stderr,
    'Avg Unmodified Returns': avg_unmodified_returns,
    'Avg Unmodified Returns Std Error': avg_unmodified_returns_sterr,
    'Avg Interventions': avg_interventions,
    'Avg Interventions Std Error': avg_interventions_sterr
})
data_to_plot.to_csv('plots/hitl_env_basic_data.csv')
