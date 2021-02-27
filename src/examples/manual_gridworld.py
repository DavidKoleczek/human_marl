# Let's you manually explore the Gridworld env

# need some boilerplate so we can import from the src folder
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.gridworld_environment import GridworldEnvironment  # pylint: disable=import-error

env = GridworldEnvironment()

env.reset()
for timesteps in range(1000):
    action = int(input('chose an action 0-3: '))

    env.step(action)

    print('state: {}, reward: {}'.format(env.state['observation'].numpy()[0], env.state['reward']))

    if env.state['done']:
        env.reset()
