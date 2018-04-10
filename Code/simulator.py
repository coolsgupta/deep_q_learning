import gym
import warnings
import os
import time
import random
import csv
import numpy as np

class simulator(object):

    def __init__(self, agent, env = "CartPole-v1", algo = "DQN", log_metrics=False, testing = False, max_episodes):
        # creating the environment
        self.env = gym.make(env)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.done = False
        self.episode = 0;
        self.testing = testing
        self.reward = 0;
        self.agent = agent
        self.EPISODES = max_episodes

        # creating log files in csv format
        self.log_metrics = log_metrics
        if self.log_metrics == True:
            self.log_fields = ['episode', 'testing', 'reward']
            self.log_filename = os.path.join("logs", self.agent.name)
            self.log_filename = os.path.join(self.log_filename, algo)
            self.log_file = open(self.log_filename, 'wb')
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.log_fields)
            self.log_writer.writeheader()

    def run(self, tolerance=0.05, n_test=0,):
        # runs a simulation of the environment
        for e in self.EPISODES:
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range (self.max_frames):
                # for Gui
                self.env.render()

                # decide action
                action = self.agent.act(state)

                # advance the game to the next frame based on the action
                next_state, reward, done = self.env.step(action)
                self.reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, self.state_size])

                # Remember the previous state, action, reward, and done
                self.agent.remember(state, action, reward, next_state, done)

                # make next_state the new current state for the next frame.
                state = next_state

                # done becomes True when the game ends
                # log data when episode ends
                if self.done and self.log_metrics :
                    self.log_writer.writerow({
                        'episode' : self.episode,
                        'testing' : self.testing,
                        'reward' : self.reward
                    })
                    break;

        self.agent.train_with_episode_experience