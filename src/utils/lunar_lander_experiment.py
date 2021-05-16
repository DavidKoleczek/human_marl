from timeit import default_timer as timer
import numpy as np
from all.experiments.writer import ExperimentWriter
from all.experiments.experiment import Experiment
from utils.lunar_lander_utils import onehot_decode
import torch
from PIL import Image
import os

class LundarLanderExperiment(Experiment):
    '''An Experiment object for training and testing agents that interact with one environment at a time.'''
    def __init__(
            self,
            agent,
            env,
            logdir='runs',
            quiet=False,
            render=False,
            write_loss=True,
            intervention_punishment = None,
            name = None,
            path = None,
            is_penalty_adapt = False
    ):
        super().__init__(self._make_writer(logdir, agent.__name__, env.name, write_loss), quiet)
        self._agent = agent(env, self._writer)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1
        self._intervention_punishment = intervention_punishment
        self._name = name
        self._path = path
        self._is_penalty_adapt = is_penalty_adapt

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
        episode_steps = []

        step = 0
        while not self._done(frames, episodes):
            rewards, outcomes, times, steps = self._run_training_episode()
            episode_rewards.append(rewards)
            episode_outcomes.append(outcomes)
            episode_steps.append(steps)
            step += 1
            if step % 100 == 0:
                self._log_100_performance(episode_rewards, episode_outcomes, episode_times, episode_steps)

    def _log_100_performance(self, episode_rewards, episode_outcomes, episode_times, episode_steps):
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        mean_100ep_succ = round(np.mean([1 if x==100 else 0 for x in episode_outcomes[-101:-1]]), 2)
        mean_100ep_crash = round(np.mean([1 if x==-100 else 0 for x in episode_outcomes[-101:-1]]), 2)
        sum_100ep_time = int(np.sum(episode_times[-101:-1]))
        num_episodes = len(episode_rewards)
        mean_100ep_step = round(np.mean(episode_steps[-101:-1]), 1)
    
        print("----------------------------------------------------------")
        print("episodes", num_episodes)
        print("mean 100 episode reward", mean_100ep_reward)
        print("mean 100 episode steps", mean_100ep_step)
        print("mean 100 episode succ", mean_100ep_succ)
        print("mean 100 episode crash", mean_100ep_crash)
        print("% time spent exploring", sum_100ep_time)
        print("----------------------------------------------------------")

    def test(self, episodes=100, policy = None):
        episode_rewards = []
        episode_outcomes = []
        episode_times = []
        episode_steps = []
        for episode in range(episodes):
            rewards, outcomes, times, steps = self._run_test_episode(policy)
            episode_rewards.append(rewards)
            episode_outcomes.append(outcomes)
            episode_times.append(times)
            episode_steps.append(steps)
            #self._log_test_episode(episode, rewards)
            print('episode: {}, rewards: {}, total steps: {}'.format(episode, rewards, steps))
        self._log_test(episode_rewards)
        self._log_100_performance(episode_rewards, episode_outcomes, episode_times, episode_steps)
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

        return returns, state['reward'], end_time - start_time, self._frame - start_frame

    def _run_test_episode(self, policy = None):
        if not policy:
        # use defalut policy
            policy = self._agent.eval

        # initialize timer
        start_time = timer()

        # initialize the episode
        steps = 0
        self._env.reset()
        state = self._env.state
        action = policy(state)
        # print("state", state)
        # print("action", action)
        returns = 0
        steps += 1

        # loop until the episode is finished
        while not state['done']:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            action = policy(state)
            # print("state", state)
            # print("action", action)
            returns += state['reward']
            steps += 1

        # stop the timer
        end_time = timer()

        return returns, state['reward'], end_time - start_time, steps

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






    def intervention_train(self, frames=np.inf, episodes=np.inf):
        episode_rewards = []
        episode_outcomes = []
        episode_times = []
        episode_interventions = []
        episode_steps = []

        best_reward = -np.inf

        step = 0
        while not self._done(frames, episodes):
            rewards, outcomes, times, interventions, steps = self._intervention_run_training_episode()
            episode_rewards.append(rewards)
            episode_outcomes.append(outcomes)
            episode_times.append(times)
            episode_interventions.append(interventions)
            episode_steps.append(steps)

            step += 1
            if step % 100 == 0:
                mean_100ep_reward = self._intervention_log_100_performance(episode_rewards, episode_outcomes, episode_times, episode_interventions, episode_steps)
                if best_reward < mean_100ep_reward:
                    best_reward = mean_100ep_reward
                    model = self._agent
                    if self._is_penalty_adapt:
                        state = {'q':model.q.model.state_dict(), 'penalty':model.penalty}
                        torch.save(state, self._path)
                    else:
                        state = {'q':model.q.model.state_dict(), 'policy.epsilon':model.policy.epsilon}
                        torch.save(state, self._path)
                    print("Best performance at the moment! Save the model.")
                print("----------------------------------------------------------")


    def _intervention_log_100_performance(self, episode_rewards, episode_outcomes, episode_times, episode_interventions, episode_steps):
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        std_100ep_reward = round(np.std(episode_rewards[-101:-1], ddof = 1), 1)
        mean_100ep_succ = round(np.mean([1 if x==100 else 0 for x in episode_outcomes[-101:-1]]), 2)
        mean_100ep_crash = round(np.mean([1 if x==-100 else 0 for x in episode_outcomes[-101:-1]]), 2)
        sum_100ep_time = int(np.sum(episode_times[-101:-1]))
        num_episodes = len(episode_rewards)
        mean_100ep_intervention = round(np.mean(episode_interventions[-101:-1]), 1)
        std_100ep_intervention = round(np.std(episode_interventions[-101:-1], ddof = 1), 1)
        mean_100ep_step = round(np.mean(episode_steps[-101:-1]), 1)
        std_100ep_step = round(np.std(episode_steps[-101:-1], ddof = 1), 1)
    
        print("----------------------------------------------------------")
        print("episodes", num_episodes)
        print("mean 100 episode reward", mean_100ep_reward)
        print("std 100 episode reward", std_100ep_reward)
        print("mean 100 episode intervention", mean_100ep_intervention)
        print("std 100 episode intervention", std_100ep_intervention)
        print("mean 100 episode steps", mean_100ep_step)
        print("std 100 episode steps", std_100ep_step)
        print("mean 100 episode succ", mean_100ep_succ)
        print("mean 100 episode crash", mean_100ep_crash)
        print("% time spent exploring", sum_100ep_time)
        if self._is_penalty_adapt:
            print("penalty", self._agent.penalty)
        

        return mean_100ep_reward

    def intervention_test(self, episodes=100, policy = None):
        if self._is_penalty_adapt:
            print("penalty", self._agent.penalty)
        f = open(self._name + ".csv",'a')
        episode_rewards = []
        episode_outcomes = []
        episode_times = []
        episode_interventions = []
        episode_steps = []
        for episode in range(episodes):
            rewards, outcomes, times, interventions, steps = self._intervention_run_test_episode(policy)
            episode_rewards.append(rewards)
            episode_outcomes.append(outcomes)
            episode_times.append(times)
            episode_interventions.append(interventions)
            episode_steps.append(steps)
            #self._log_test_episode(episode, rewards)

            succ = 0
            crash = 0
            if outcomes == 100:
                succ = 1
            elif outcomes == -100:
                crash = 1
            print('episode: {}, rewards: {}, interventions: {}, total steps: {}, succ: {}, crash: {}'.format(episode, rewards, interventions, steps, succ, crash))
            f.write(str(rewards) + ", " + str(interventions) + ", " + str(steps) + ", " + str(succ) + ", " + str(crash) + "\n" ) 
        f.close()
        self._log_test(episode_rewards)
        self._intervention_log_100_performance(episode_rewards, episode_outcomes, episode_times, episode_interventions, episode_steps)
        return episode_rewards, episode_outcomes, episode_interventions, episode_steps

    def _intervention_run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        interventions = 0
        succ = 0
        crash = 0

        # initialize the episode
        self._env.reset()
        state = self._env.state
        pilot_action = onehot_decode(state.observation[-self._env.action_space.n:])
        action = None
        if self._intervention_punishment:
            action = self._agent.act(state, self._env.action_space.n, self._intervention_punishment)
        else:
            action = self._agent.act(state)
        returns = 0

        if action != pilot_action:
            interventions += 1


        # loop until the episode is finished
        while not state['done']:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            pilot_action = onehot_decode(state.observation[-self._env.action_space.n:]) 
            action = None
            if self._intervention_punishment:
                action = self._agent.act(state, self._env.action_space.n, self._intervention_punishment)
            else:
                action = self._agent.act(state)
            returns += state['reward']
            self._frame += 1
        
            if action != pilot_action:
                interventions += 1

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        steps = self._frame - start_frame


        if state['reward'] == 100:
            succ = 1
        elif state['reward'] == -100:
            crash = 1


        # log the results
        #self._log_training_episode(returns, fps)
        print('episode: {}, frame: {}, fps: {}, returns: {}, interventions: {}, steps: {}, succ: {}, crash: {}'.format(self._episode, self._frame, int(fps), returns, interventions, steps, succ, crash))

        # update experiment state
        self._episode += 1

        return returns, state['reward'], end_time - start_time, interventions, steps

    def _intervention_run_test_episode(self, policy = None):
        if not policy:
        # use defalut policy
            policy = self._agent.eval

        # initialize timer
        start_time = timer()

        interventions = 0
        steps = 0

        # initialize the episode
        self._env.reset()
        state = self._env.state
        pilot_action = onehot_decode(state.observation[-self._env.action_space.n:]) 
        action = policy(state)
        returns = 0
        steps += 1

        if action != pilot_action:
            interventions += 1

        # loop until the episode is finished
        while not state['done']:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            pilot_action = onehot_decode(state.observation[-self._env.action_space.n:]) 
            action = policy(state)
            returns += state['reward']
            steps += 1

            if action != pilot_action:
                interventions += 1

        # stop the timer
        end_time = timer()

        return returns, state['reward'], end_time - start_time, interventions, steps

    def intervention_show(self, policy = None):
        render = self._render
        self._render = True
        self._intervention_run_test_episode(policy)
        self._env.close()
        self._render = render


    def observe_interention(self, policy = None, image_path = None):
        render = self._render
        self._render = True
        self._observe_test_episode(policy, image_path)
        self._env.close()
        self._render = render

    def _observe_test_episode(self, policy = None, image_path = None):
        if not policy:
        # use defalut policy
            policy = self._agent.eval

        # initialize timer
        start_time = timer()

        interventions = 0
        steps = 0

        # initialize the episode
        self._env.reset()
        state = self._env.state
        pilot_action = onehot_decode(state.observation[-self._env.action_space.n:]) 
        action = policy(state)

        returns = 0
        steps += 1

        print("steps", steps)
        print("state", state)
        print("action", action)

        if action != pilot_action:
            interventions += 1
            print("intervene")
            print(state.observation)

        # loop until the episode is finished
        while not state['done']:
            if self._render:
                frame = self._env.render(mode='rgb_array')
                im = Image.fromarray(frame)
                name = str(steps) + ".png"
                path = os.path.join(image_path, name)

                if not os.path.exists(image_path):
                    os.makedirs(image_path)

                im.save(path)

            state = self._env.step(action)
            pilot_action = onehot_decode(state.observation[-self._env.action_space.n:]) 
            action = policy(state)
            print("steps", steps)
            print("state", state)
            print("action", action)
            returns += state['reward']
            steps += 1

            if action != pilot_action:
                interventions += 1
                print(state.observation.numpy())
                print("intervene!")
            print("pilot action: " + str(pilot_action) + " copilot action: " + str(action))
            print("-------------------------------------------------")

        # stop the timer
        end_time = timer()

        return returns, state['reward'], end_time - start_time, interventions, steps


        

    def record_observation(self, policy = None):
        intervention_name = "observation_intervention_" + self._name + ".csv"
        f = open(intervention_name,'a')

        full_name = "observation_full_" + self._name + ".csv"
        full_f = open(full_name,'a')
        if not policy:
        # use defalut policy
            policy = self._agent.eval

        # initialize timer
        start_time = timer()

        interventions = 0
        steps = 0

        # initialize the episode
        self._env.reset()
        state = self._env.state
        pilot_action = onehot_decode(state.observation[-self._env.action_space.n:]) 
        action = policy(state)

        returns = 0
        steps += 1


        obs = ""
        cnt = 0
        for i in state.observation.numpy():
            cnt += 1
            if cnt == state.observation.numpy().shape[0]:
                obs += str(i) + "\n"
            else:
                obs += str(i) + ", "
        full_f.write(obs) 

        # print("steps", steps)
        # print("state", state)
        # print("action", action)

        if action != pilot_action:
            interventions += 1
            #print("intervene")
            #print(state.observation)
            intervention_obs = ""
            cnt = 0
            for i in state.observation.numpy():
                cnt += 1
                if cnt == state.observation.numpy().shape[0]:
                    intervention_obs += str(i) + "\n"
                else:
                    intervention_obs += str(i) + ", "
            f.write(intervention_obs) 
            #print(intervention_obs)
        

        # loop until the episode is finished
        while not state['done']:
            # if self._render:
            #     frame = self._env.render(mode='rgb_array')
            #     im = Image.fromarray(frame)
            #     name = str(steps) + ".png"
            #     path = os.path.join(image_path, name)

            #     if not os.path.exists(image_path):
            #         os.makedirs(image_path)

            #     im.save(path)

            state = self._env.step(action)
            pilot_action = onehot_decode(state.observation[-self._env.action_space.n:]) 
            action = policy(state)
            # print("steps", steps)
            # print("state", state)
            # print("action", action)
            returns += state['reward']
            steps += 1

            obs = ""
            cnt = 0
            for i in state.observation.numpy():
                cnt += 1
                if cnt == state.observation.numpy().shape[0]:
                    obs += str(i) + "\n"
                else:
                    obs += str(i) + ", "
            full_f.write(obs) 

            if action != pilot_action:
                interventions += 1
                #print(state.observation.numpy())
                #f.write(str(state.observation.numpy()))
                intervention_obs = ""
                cnt = 0
                for i in state.observation.numpy():
                    cnt += 1
                    if cnt == state.observation.numpy().shape[0]:
                        intervention_obs += str(i) + "\n"
                    else:
                        intervention_obs += str(i) + ", "
                #print(intervention_obs)
                f.write(intervention_obs) 
            #     print("intervene!")
            # print("pilot action: " + str(pilot_action) + " copilot action: " + str(action))
            # print("-------------------------------------------------")

        # stop the timer
        end_time = timer()
        f.close()

        return returns, state['reward'], end_time - start_time, interventions, steps















    def budget_train(self, frames=np.inf, episodes=np.inf):
        episode_rewards = []
        episode_outcomes = []
        episode_times = []
        episode_interventions = []
        episode_steps = []

        best_reward = -np.inf

        step = 0
        while not self._done(frames, episodes):
            rewards, outcomes, times, interventions, steps = self._budget_run_training_episode()
            episode_rewards.append(rewards)
            episode_outcomes.append(outcomes)
            episode_times.append(times)
            episode_interventions.append(interventions)
            episode_steps.append(steps)

            step += 1
            if step % 100 == 0:
                mean_100ep_reward = self._budget_log_100_performance(episode_rewards, episode_outcomes, episode_times, episode_interventions, episode_steps)
                if best_reward < mean_100ep_reward:
                    best_reward = mean_100ep_reward
                    model = self._agent
                    state = {'q':model.q.model.state_dict(), 'policy.epsilon':model.policy.epsilon}
                    torch.save(state, self._path)
                    print("Best performance at the moment! Save the model.")
                print("----------------------------------------------------------")


    def _budget_log_100_performance(self, episode_rewards, episode_outcomes, episode_times, episode_interventions, episode_steps):
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        std_100ep_reward = round(np.std(episode_rewards[-101:-1], ddof = 1), 1)
        mean_100ep_succ = round(np.mean([1 if x==100 else 0 for x in episode_outcomes[-101:-1]]), 2)
        mean_100ep_crash = round(np.mean([1 if x==-100 else 0 for x in episode_outcomes[-101:-1]]), 2)
        sum_100ep_time = int(np.sum(episode_times[-101:-1]))
        num_episodes = len(episode_rewards)
        mean_100ep_intervention = round(np.mean(episode_interventions[-101:-1]), 1)
        std_100ep_intervention = round(np.std(episode_interventions[-101:-1], ddof = 1), 1)
        mean_100ep_step = round(np.mean(episode_steps[-101:-1]), 1)
        std_100ep_step = round(np.std(episode_steps[-101:-1], ddof = 1), 1)
    
        print("----------------------------------------------------------")
        print("episodes", num_episodes)
        print("mean 100 episode reward", mean_100ep_reward)
        print("std 100 episode reward", std_100ep_reward)
        print("mean 100 episode intervention", mean_100ep_intervention)
        print("std 100 episode intervention", std_100ep_intervention)
        print("mean 100 episode steps", mean_100ep_step)
        print("std 100 episode steps", std_100ep_step)
        print("mean 100 episode succ", mean_100ep_succ)
        print("mean 100 episode crash", mean_100ep_crash)
        print("% time spent exploring", sum_100ep_time)
        

        return mean_100ep_reward

    def budget_test(self, episodes=100, policy = None):
        f = open(self._name + ".csv",'a')
        episode_rewards = []
        episode_outcomes = []
        episode_times = []
        episode_interventions = []
        episode_steps = []
        for episode in range(episodes):
            rewards, outcomes, times, interventions, steps = self._budget_run_test_episode(policy)
            episode_rewards.append(rewards)
            episode_outcomes.append(outcomes)
            episode_times.append(times)
            episode_interventions.append(interventions)
            episode_steps.append(steps)
            #self._log_test_episode(episode, rewards)

            succ = 0
            crash = 0
            if outcomes == 100:
                succ = 1
            elif outcomes == -100:
                crash = 1
            print('episode: {}, rewards: {}, interventions: {}, total steps: {}, succ: {}, crash: {}'.format(episode, rewards, interventions, steps, succ, crash))
            f.write(str(rewards) + ", " + str(interventions) + ", " + str(steps) + ", " + str(succ) + ", " + str(crash) + "\n" ) 
        f.close()
        self._log_test(episode_rewards)
        self._budget_log_100_performance(episode_rewards, episode_outcomes, episode_times, episode_interventions, episode_steps)
        return episode_rewards, episode_outcomes, episode_interventions, episode_steps

    def _budget_run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        interventions = 0
        succ = 0
        crash = 0

        # initialize the episode
        self._env.reset()
        state = self._env.state
        # print("state", state)
        pilot_action = onehot_decode(state.observation[-self._env.action_space.n-1:-1]) 
        budget_run_out = self._env.remaining_budget <= 0
        action = None
        if self._intervention_punishment:
            action = self._agent.act(state, self._env.action_space.n, self._intervention_punishment, budget_run_out)
        else:
            action = self._agent.act(state)
        returns = 0

        if action != pilot_action:
            if budget_run_out:
                action = pilot_action
            else:
                self._env.buget_decrease()
                interventions += 1
            

        # loop until the episode is finished
        while not state['done']:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            # print("state", state)
            pilot_action = onehot_decode(state.observation[-self._env.action_space.n-1:-1]) 
            budget_run_out = self._env.remaining_budget <= 0
            action = None
            if self._intervention_punishment:
                action = self._agent.act(state, self._env.action_space.n, self._intervention_punishment, budget_run_out)
            else:
                action = self._agent.act(state)
            returns += state['reward']
            self._frame += 1
        
            if action != pilot_action:
                if budget_run_out:
                    action = pilot_action
                else:
                    self._env.buget_decrease()
                    interventions += 1

        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        steps = self._frame - start_frame


        if state['reward'] == 100:
            succ = 1
        elif state['reward'] == -100:
            crash = 1


        # log the results
        #self._log_training_episode(returns, fps)
        print('episode: {}, frame: {}, fps: {}, returns: {}, interventions: {}, steps: {}, succ: {}, crash: {}'.format(self._episode, self._frame, int(fps), returns, interventions, steps, succ, crash))

        # update experiment state
        self._episode += 1

        return returns, state['reward'], end_time - start_time, interventions, steps

    def _budget_run_test_episode(self, policy = None):
        if not policy:
        # use defalut policy
            policy = self._agent.eval

        # initialize timer
        start_time = timer()

        interventions = 0
        steps = 0

        # initialize the episode
        self._env.reset()
        state = self._env.state
        pilot_action = onehot_decode(state.observation[-self._env.action_space.n-1:-1])  
        budget_run_out = self._env.remaining_budget <= 0
        action = policy(state)
        returns = 0
        steps += 1

        if action != pilot_action:
            if budget_run_out:
                action = pilot_action
            else:
                self._env.buget_decrease()
                interventions += 1

        # loop until the episode is finished
        while not state['done']:
            if self._render:
                self._env.render()
            state = self._env.step(action)
            pilot_action = onehot_decode(state.observation[-self._env.action_space.n-1:-1]) 
            budget_run_out = self._env.remaining_budget <= 0 
            action = policy(state)
            returns += state['reward']
            steps += 1


            if action != pilot_action:
                if budget_run_out:
                    action = pilot_action
                else:
                    self._env.buget_decrease()
                    interventions += 1

        # stop the timer
        end_time = timer()

        return returns, state['reward'], end_time - start_time, interventions, steps

    def budget_show(self, policy = None):
        render = self._render
        self._render = True
        self._budget_run_test_episode(policy)
        self._env.close()
        self._render = render




    def budget_record_observation(self, policy = None):
        intervention_name = "observation_intervention_" + self._name + ".csv"
        f = open(intervention_name,'a')

        full_name = "observation_full_" + self._name + ".csv"
        full_f = open(full_name,'a')
        if not policy:
        # use defalut policy
            policy = self._agent.eval

        # initialize timer
        start_time = timer()

        interventions = 0
        steps = 0

        # initialize the episode
        self._env.reset()
        state = self._env.state
        pilot_action = onehot_decode(state.observation[-self._env.action_space.n-1:-1])  
        budget_run_out = self._env.remaining_budget <= 0
        action = policy(state)
        returns = 0
        steps += 1


        obs = ""
        cnt = 0
        for i in state.observation.numpy():
            cnt += 1
            if cnt == state.observation.numpy().shape[0]:
                obs += str(i) + "\n"
            else:
                obs += str(i) + ", "
        full_f.write(obs) 


        if action != pilot_action:
            if budget_run_out:
                action = pilot_action
            else:
                self._env.buget_decrease()
                interventions += 1
                interventions += 1
                intervention_obs = ""
                cnt = 0
                for i in state.observation.numpy():
                    cnt += 1
                    if cnt == state.observation.numpy().shape[0]:
                        intervention_obs += str(i) + "\n"
                    else:
                        intervention_obs += str(i) + ", "
                f.write(intervention_obs) 
            #print(intervention_obs)
        

        # loop until the episode is finished
        while not state['done']:
            state = self._env.step(action)
            pilot_action = onehot_decode(state.observation[-self._env.action_space.n-1:-1]) 
            budget_run_out = self._env.remaining_budget <= 0 
            action = policy(state)
            returns += state['reward']
            steps += 1

            obs = ""
            cnt = 0
            for i in state.observation.numpy():
                cnt += 1
                if cnt == state.observation.numpy().shape[0]:
                    obs += str(i) + "\n"
                else:
                    obs += str(i) + ", "
            full_f.write(obs) 

            if action != pilot_action:
                if budget_run_out:
                    action = pilot_action
                else:
                    self._env.buget_decrease()
                    interventions += 1
                    interventions += 1
                    intervention_obs = ""
                    cnt = 0
                    for i in state.observation.numpy():
                        cnt += 1
                        if cnt == state.observation.numpy().shape[0]:
                            intervention_obs += str(i) + "\n"
                        else:
                            intervention_obs += str(i) + ", "
                    f.write(intervention_obs) 

        # stop the timer
        end_time = timer()
        f.close()

        return returns, state['reward'], end_time - start_time, interventions, steps