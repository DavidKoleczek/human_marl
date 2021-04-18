''' Run this script to play LunarLander yourself with a "human in the loop" agent helping you!

To try other trained agents, change the variable HITL_LUNAR_AGENT_PATH near the top of this file.

Requires:
- Stable Baselines 3 https://github.com/DLR-RM/stable-baselines3#installation
- Pygame https://www.pygame.org/wiki/GettingStarted
- PyTorch
- OpenAI Gym (also requires Box2D)

Reference implementation: https://github.com/openai/gym/blob/master/gym/utils/play.py
'''

import argparse

import pygame
from pygame.locals import VIDEORESIZE
from stable_baselines3 import SAC
import numpy as np
import pandas as pd

from environments.hitl_sb_lunarlandercont_eval import HITLSBLunarLanderContEval


HITL_LUNAR_AGENT_PATH = 'savedModels/sac_lunar_hitl_1p_sensor00.zip'


def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def _add_intervention_marker(display_pixels):
    x_axis = list(range(500, 530))
    y_axis = list(range(50, 80))
    for x in x_axis:
        for y in y_axis:
            display_pixels[y][x] = (255, 30, 30)

    return display_pixels


def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    """Allows one to play the game using keyboard.
    To simply play the game use:
        play(gym.make("Pong-v4"))
    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.

    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environment is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:
            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """
    env.reset()
    rendered = env.render(mode='rgb_array')

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

    video_size = [rendered.shape[1], rendered.shape[0]]
    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    dataset = []  # log dataset
    episode_counter = 0  # log dataset
    while running:
        if env_done:
            env_done = False
            obs = env.reset()
            info = {}
            episode_counter += 1
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)
            dataset.append(np.append(obs, [episode_counter]))  # log dataset
            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info)
        if obs is not None:
            # TODO: If we make an interventions, flash that onto the screen by modifying the raw pixels.
            rendered = env.render(mode='rgb_array')
            if info and info['did_intervene']:
                rendered = _add_intervention_marker(rendered)
            display_arr(screen, rendered, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
    dataset = pd.DataFrame(dataset, columns=['Pos Horizontal', 'Pos Vertical', 'Speed Horizontal', 'Speed Vertical', 'Angle',
                                             'Angular Speed', 'Leg 1 Contact', 'Leg 2 Contact', 'Main Engine Action', 'Left-Right Action', 'Was Intervention', 'Episode Number'])
    dataset.to_csv('plots/data_human_sac_lunar_hitl_1p_sensor01.csv', index=False)


def print_rewards_callback(obs_t, obs_tp1, action, rew, done, info):
    return print(rew)


def main(trained_agent_type, zoom_level):
    # mapping lunar lander controls to "W" (main engine), "A" (left engine), "D" (right engine)
    keys_to_action = {
        (ord('w'), ): 2,
        (ord('a'), ): 1,
        (ord('d'), ): 3,
        (ord('d'), ord('w')): 3,
        (ord('a'), ord('w')): 1,
    }

    # Checking for various trained_agent_type that might be selected.
    # 0: The human has full control.
    # 1: Trained with Sensor human and intervention penalty of 1
    # 2: Trained with Noisy human and intervention penalty of 0.15
    # 3: Trained with Noisy human and intervention penalty of 0.75
    # 4: Ensemble of 1, 2, and 3. i.e. an action is sampled uniformly randomly from one of those agents at each timestep
    if trained_agent_type == 0:
        # this agent doesn't actually do anything, just a placeholder to satisfy HITLSBLunarLanderContEval's API
        hitl_agent = SAC.load('sac_lunar_hitl_1p_sensor00.zip')
        eval_env = HITLSBLunarLanderContEval('LunarLanderContinuous-v2', hitl_agent, do_not_intervene=True)
        play(eval_env, zoom=zoom_level, fps=60, keys_to_action=keys_to_action, callback=print_rewards_callback)
    elif trained_agent_type == 4:
        hitl_agent1 = SAC.load('savedModels/sac_lunar_hitl_1p_sensor00.zip')
        hitl_agent2 = SAC.load('savedModels/sac_lunar_hitl_015p_noisy085.zip')
        hitl_agent3 = SAC.load('savedModels/sac_lunar_hitl_075p_noisy085.zip')
        eval_env = HITLSBLunarLanderContEval('LunarLanderContinuous-v2', [hitl_agent1, hitl_agent2, hitl_agent3])
        play(eval_env, zoom=zoom_level, fps=60, keys_to_action=keys_to_action, callback=print_rewards_callback)
    else:
        if trained_agent_type == 1:
            HITL_LUNAR_AGENT_PATH = 'savedModels/sac_lunar_hitl_1p_sensor00.zip'
        elif trained_agent_type == 2:
            HITL_LUNAR_AGENT_PATH = 'savedModels/sac_lunar_hitl_015p_noisy085.zip'
        else:
            HITL_LUNAR_AGENT_PATH = 'savedModels/sac_lunar_hitl_075p_noisy085.zip'

        # load a saved human in the loop agent for LunarLander
        hitl_agent = SAC.load(HITL_LUNAR_AGENT_PATH)
        # create an instance of an evaluation environment, which takes in human actions in its "step" function
        eval_env = HITLSBLunarLanderContEval('LunarLanderContinuous-v2', hitl_agent)
        play(eval_env, zoom=zoom_level, fps=60, keys_to_action=keys_to_action, callback=print_rewards_callback)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_agent_type', type=int, default=1)
    parser.add_argument('--zoom_level', type=int, default=2)

    args = vars(parser.parse_args())
    main(args['trained_agent_type'], args['zoom_level'])
