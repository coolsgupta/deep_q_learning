import random
import gym
from collections import deque
import math
#from .evaluation import evaluate
import json
EPISODES = 1000
import matplotlib.pyplot as plt
import csv
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.Q_table = dict()
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.t = 0
        self.learning = True
        self.episode = 0



    def update(self, testing = False):
        if testing:
            self.epsilon = 0.0
            self.learning_rate = 0.0
        else:
            self.epsilon = math.exp(-self.learning_rate * (self.episode))

    def build_state(self, observed_state):
        state = (str(observed_state[0]),str(observed_state[1]),str(observed_state[2]),str(observed_state[3]))
#        state = (''.join(observed_state[0]),''.join(observed_state[1]),''.join(observed_state[2]),''.join(observed_state[3]))
        state = ''.join(state)
        return state

    def get_maxQ(self,state):
        if not state in self.Q_table:
            self.createQ(state)
        maxQ = max(self.Q_table[state].values())
        max_act = []
        for act,val in self.Q_table[state].items():
            if val == maxQ:
                max_act.append(act)
        return maxQ, max_act

    def createQ(self, state):
        if self.learning:
            if not state in self.Q_table:
                self.Q_table[state] = dict((act, 0.0) for act in range(self.action_size))


    def choose_action(self, state):
        if not self.learning or random.random() <= self.epsilon:
            action = random.choice(range(self.action_size))
        else:
            maxQ, maxQ_action = self.get_maxQ(state)
            action = random.choice(maxQ_action)
        return action

    def learn(self, state, action, reward, next_state):
        if self.learning:
            maxQ, maxQ_action = self.get_maxQ(next_state)
            self.Q_table[state][action] = self.Q_table[state][action] + (self.learning_rate*(reward + (self.gamma*maxQ)))


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



    def replay(self):
        self.episode += 1
        for state, action, reward, next_state, done in self.memory:
            self.learn(state,action,reward,next_state)
        if self.epsilon > self.epsilon_min:
            self.update()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


#if __name__ == "__main__":

# initialize gym environment and the agent
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32
scores = []
#storing results
result_file = open('results_Q_cartpole.txt','w')
result_csv = open('results_Q_cartpole.csv', 'w',newline='')
fieldnames = ['episode', 'epsilon', 'score', 'average_score', 'total_reward', 'average_reward']
result_writer = csv.DictWriter(result_csv, fieldnames)
result_writer.writeheader()
cummulative_score = 0
cummulative_reward = 0

# Iterate the game
agent.episode = 1

while agent.epsilon > agent.epsilon_min:
    # reset state in the beginning of each game
    state = agent.build_state(env.reset())
    total_reward = 0
    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    for time in range(500):
        #for GUI
        #env.render()

        #adding state to Q-table
        agent.createQ(state)

        #Decide action
        action = agent.choose_action(state)

        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        total_reward += reward
        next_state = agent.build_state(next_state)

        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)

        #updating Q-table
        agent.learn(state,action,reward,next_state)
        # make next_state the new current state for the next frame.
        state = next_state

        # done becomes True when the game ends
        # ex) The agent drops the pole
        if done:
            scores.append(time)
            # print the score and break out of the loop
            cummulative_reward += total_reward
            average_reward = cummulative_reward / (agent.episode + 1)
            cummulative_score += time
            average_score = cummulative_score / (agent.episode + 1)
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(agent.episode, EPISODES, time, agent.epsilon))
            result = ('Episode :', str(agent.episode), ' score:',str(time), ' epsilon:', str(agent.epsilon),'\n')
            result_file.write(''.join(result))
            result_writer.writerow(
                {'episode': agent.episode, 'epsilon': agent.epsilon, 'score': time, 'average_score': average_score,
                 'total_reward': total_reward, 'average_reward': average_reward})

            break

    agent.replay()
# store Q_table as txt in json format
agent.Q_table = {'Q_table': agent.Q_table}
with open('Q_table_Q_cartpole.txt', 'w+') as Q_table_file:
    Q_table_file.write(json.dumps(agent.Q_table))

# testing trials
print("///////////TESTING/////////")
agent.epsilon = 0.0
cummulative_score = 0
cummulative_reward = 0
test_results = open('test_results_Q_learning_agent.csv', 'w',newline='')
test_result_writer = csv.DictWriter(test_results,fieldnames)
test_result_writer.writeheader()
for e in range(50):
    state = agent.build_state(env.reset())
    total_reward = 0
    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    for time in range(500):
        # for GUI
        # env.render()

        # Decide action
        action = agent.choose_action(state)

        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        total_reward += reward
        next_state = agent.build_state(next_state)

        # make next_state the new current state for the next frame.
        state = next_state

        # done becomes True when the game ends
        # ex) The agent drops the pole
        if done:
            # print and store the metrics and break out of the loop
            cummulative_reward += total_reward
            average_reward = cummulative_reward / (e+1)
            cummulative_score += time
            average_score = cummulative_score / (e+1)
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, time, agent.epsilon))
            test_result_writer.writerow({'episode':e, 'epsilon':agent.epsilon, 'score':time,'average_score':average_score,
                                    'total_reward': total_reward, 'average_reward':average_reward})
            break
