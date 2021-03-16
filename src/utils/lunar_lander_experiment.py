from timeit import default_timer as timer
import numpy as np
from all.experiments.writer import ExperimentWriter
from all.experiments.experiment import Experiment

class LundarLanderExperiment(Experiment):
    '''An Experiment object for training and testing agents that interact with one environment at a time.'''
    def __init__(
            self,
            agent,
            env,
            logdir='runs',
            quiet=False,
            render=False,
            write_loss=True
    ):
        super().__init__(self._make_writer(logdir, agent.__name__, env.name, write_loss), quiet)
        self._agent = agent(env, self._writer)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1

        if render:
            self._env.render(mode="human")

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def train(self, frames=np.inf, episodes=np.inf):
        episode_rewards = []
        episode_outcomes = []
        episode_times = []

        step = 0
        while not self._done(frames, episodes):
            rewards, outcomes, times = self._run_training_episode()
            episode_rewards.append(rewards)
            episode_outcomes.append(outcomes)
            episode_times.append(times)
            step += 1
            if step % 100 == 0:
                self._log_100_performance(episode_rewards, episode_outcomes, episode_times)

    def _log_100_performance(self, episode_rewards, episode_outcomes, episode_times):
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        mean_100ep_succ = round(np.mean([1 if x==100 else 0 for x in episode_outcomes[-101:-1]]), 1)
        mean_100ep_crash = round(np.mean([1 if x==-100 else 0 for x in episode_outcomes[-101:-1]]), 1)
        sum_100ep_time = int(np.sum(episode_times[-101:-1]))
        num_episodes = len(episode_rewards)
    
        print("----------------------------------------------------------")
        print("episodes", num_episodes)
        print("mean 100 episode reward", mean_100ep_reward)
        print("mean 100 episode succ", mean_100ep_succ)
        print("mean 100 episode crash", mean_100ep_crash)
        print("% time spent exploring", sum_100ep_time)
        print("----------------------------------------------------------")

    # def test(self, episodes=100):
    #     episode_rewards = []
    #     episode_outcomes = []
    #     episode_times = []
    #     for episode in range(episodes):
    #         rewards, outcomes, times = self._run_test_episode()
    #         episode_rewards.append(rewards)
    #         episode_outcomes.append(outcomes)
    #         episode_times.append(times)
    #         self._log_test_episode(episode, rewards)
    #     self._log_test(episode_rewards)
    #     self._log_100_performance(episode_rewards, episode_outcomes, episode_times)
    #     return episode_rewards

    def test(self, episodes=100, policy = None):
        episode_rewards = []
        episode_outcomes = []
        episode_times = []
        for episode in range(episodes):
            rewards, outcomes, times = self._run_test_episode(policy)
            episode_rewards.append(rewards)
            episode_outcomes.append(outcomes)
            episode_times.append(times)
            self._log_test_episode(episode, rewards)
        self._log_test(episode_rewards)
        self._log_100_performance(episode_rewards, episode_outcomes, episode_times)
        return episode_rewards

    def _run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # initialize the episode
        self._env.reset()
        state = self._env.state
        action = self._agent.act(state)
        returns = 0

        # loop until the episode is finished
        while not state['done']:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = self._agent.act(state)
            returns += state['reward']
            self._frame += 1

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        # log the results
        self._log_training_episode(returns, fps)

        # update experiment state
        self._episode += 1

        return returns, state['reward'], end_time - start_time

    # def _run_test_episode(self, policy = self._agent.eval):
    #     # initialize timer
    #     start_time = timer()

    #     # initialize the episode
    #     self._env.reset()
    #     state = self._env.state
    #     action = self._agent.eval(state)
    #     returns = 0

    #     # loop until the episode is finished
    #     while not state['done']:
    #         if self._render:
    #             self._env.render()
    #         state = self._env.step(action)
    #         action = self._agent.eval(state)
    #         returns += state['reward']

    #     # stop the timer
    #     end_time = timer()

    #     return returns, state['reward'], end_time - start_time


    def _run_test_episode(self, policy = None):
        if not policy:
        # use defalut policy
            policy = self._agent.eval

        # initialize timer
        start_time = timer()

        # initialize the episode
        self._env.reset()
        state = self._env.state
        action = policy(state)
        returns = 0

        # loop until the episode is finished
        while not state['done']:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = policy(state)
            returns += state['reward']

        # stop the timer
        end_time = timer()

        return returns, state['reward'], end_time - start_time

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _make_writer(self, logdir, agent_name, env_name, write_loss):
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss, logdir=logdir)

    def show(self, policy = None):
        render = self._render
        self._render = True
        self._run_test_episode(policy)
        self._env.close()
        self._render = render
