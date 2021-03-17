'''Trying traditional RL algorithms on our Human in the Loop setup for modified 687 Gridworld.
'''

from easyrl.environments import GridworldEnvironment
from easyrl.agents import vac_preset
from easyrl.train.single_env_experiment import SingleEnvExperiment
from easyrl.utils import average_policy_returns
from agents.lunar_lander_simulated_agent import LaggyPilotPolicyAgent
from environments.hitl_gridworld import HITLGridworldEnvironment


env = GridworldEnvironment()

# create a fake human agent
human_performance = 0
# keep trying VAC training until we get a good agent, sometimes it takes forever to converge
while human_performance < 3:
    fake_human_agent = vac_preset(env)
    trainer = SingleEnvExperiment(fake_human_agent, env)
    trainer.train(episodes=200)
    human_performance = average_policy_returns(fake_human_agent, env, num_episodes=50)

fake_human_agent = LaggyPilotPolicyAgent(fake_human_agent)
human_performance = average_policy_returns(fake_human_agent, env, num_episodes=50)
print('Fake human policy average performance (reward): ', human_performance)


# this will be our optimally trained agent for the new environment
hitl_env = HITLGridworldEnvironment(fake_human_agent)
agent = vac_preset(hitl_env)

experiment = SingleEnvExperiment(agent, hitl_env)
experiment.train(episodes=300)

result = average_policy_returns(agent, hitl_env, num_episodes=50)
print(result)
