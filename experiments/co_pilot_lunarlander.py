import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from src.agents.ddqn_agent import DDQN_agent
from src.agents.co_ddqn_agent import co_DDQN_agent
from src.models.models import lunar_lander_nature_ddqn
from src.utils.lunar_lander_experiment import LundarLanderExperiment
from src.environments.lunar_lander_environment import make_co_env
from src.environments.lunar_lander_environment import make_env
from src.agents.lunar_lander_simulated_agent import sensor_pilot_policy, noop_pilot_policy, NoisyPilotPolicy, LaggyPilotPolicy

# dims for action and observation
n_act_dim = 6
n_obs_dim = 9

# Every episode is at most 1000 steps. Use 500 episodes to train
max_ep_len = 1000
n_training_episodes = 500
max_timesteps = max_ep_len * n_training_episodes

env = make_env(using_lander_reward_shaping=True)

agent = DDQN_agent(
    device="cpu",
    discount_factor=0.99,
    last_frame=max_timesteps,
    lr=1e-3,
    target_update_frequency=1500,
    update_frequency=1,
    final_exploration=0.02,
    final_exploration_frame=0.1 * max_timesteps,
    replay_start_size=1000,
    replay_buffer_size=50000,
    model_constructor=lunar_lander_nature_ddqn)

frames = max_timesteps

exp_pilot = LundarLanderExperiment(
    agent,
    env,
    logdir='runs',
    quiet=False,
    render=False,
    write_loss=True
)

PATH = "./saved_models/pilot_model.pkl"

load_pretrained_pilot = True

if load_pretrained_pilot:
    checkpoint = torch.load(PATH)
    exp_pilot._agent.q.model.load_state_dict(checkpoint['q'])
    exp_pilot._agent.policy.q.model.load_state_dict(checkpoint['q'])
    exp_pilot._agent.policy.epsilon = checkpoint['policy.epsilon']
else:
    exp_pilot.train(frames=frames)
    model = exp_pilot._agent
    state = {'q': model.q.model.state_dict(), 'policy.epsilon': model.policy.epsilon}
    torch.save(state, PATH)

# exp_pilot.show()


alpha = 0.2
pilot_name = "noisy_pilot"

# pilot_policy = exp_pilot._agent.policy
pilot_policy = NoisyPilotPolicy(exp_pilot._agent.policy)
# pilot_policy = LaggyPilotPolicy(exp_pilot._agent.policy)
# pilot_policy = noop_pilot_policy
#pilot_policy = sensor_pilot_policy

load_pretrained_co_pilot = False
PATH = "./saved_models/intervention_penalty/" + pilot_name + "_alpha_" + str(alpha) + ".pkl"
#PATH = "saved_models/" + pilot_name + "_0.5_alpha_" + str(alpha) + ".pkl"

print(PATH)

co_env = make_co_env(pilot_policy=pilot_policy, using_lander_reward_shaping=True)

co_agent = co_DDQN_agent(
    device="cpu",
    discount_factor=0.99,
    last_frame=max_timesteps,
    lr=1e-3,
    target_update_frequency=1500,
    update_frequency=1,
    final_exploration=0.02,
    final_exploration_frame=0.1 * max_timesteps,
    replay_start_size=1000,
    replay_buffer_size=50000,
    model_constructor=lunar_lander_nature_ddqn,
    pilot_tol=alpha
)

frames = max_timesteps

exp_co_pilot = LundarLanderExperiment(
    co_agent,
    co_env,
    logdir='runs',
    quiet=False,
    render=False,
    write_loss=True,
)

if load_pretrained_co_pilot:
    checkpoint = torch.load(PATH)
    exp_co_pilot._agent.q.model.load_state_dict(checkpoint['q'])
    exp_co_pilot._agent.policy.q.model.load_state_dict(checkpoint['q'])
    exp_co_pilot._agent.policy.epsilon = checkpoint['policy.epsilon']
else:
    exp_co_pilot.train(frames=frames)
    model = exp_co_pilot._agent
    state = {'q': model.q.model.state_dict(), 'policy.epsilon': model.policy.epsilon}
    torch.save(state, PATH)

exp_co_pilot.show()
exp_co_pilot.test()
