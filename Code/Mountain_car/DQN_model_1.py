import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import csv
EPISODES = 500


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.model1.hdf5',
                               verbose=0, save_best_only=True)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[agent.checkpointer])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


#if __name__ == "__main__":

# initialize gym environment and the agent
env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
# agent.load("./save/cartpole-dqn.h5")
done = False
batch_size = 32
# Iterate the game
#storing results
result_csv = open('results_DQN_model_1.csv','w',newline='')
fieldnames = ['episode', 'epsilon', 'score', 'average_score', 'total_reward', 'average_reward']
result_writer = csv.DictWriter(result_csv, fieldnames)
result_writer.writeheader()
cummulative_score = 0
cummulative_reward = 0
for e in range(EPISODES):
    # reset state in the beginning of each game
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    score = state[0][0]
    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    for time in range(500):
        #for GUI
        #env.render()

        #Decide action
        action = agent.act(state)

        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        total_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        score = next_state[0][0] if next_state[0][0]>score else score

        # Remember the previous state, action, reward, and done
        agent.remember(state, action, next_state[0][0], next_state, done)

        # make next_state the new current state for the next frame.
        state = next_state

        # done becomes True when the game ends
        # ex) The agent drops the pole
        if done:
            # print and store the metrics and break out of the loop
            cummulative_reward += total_reward
            average_reward = cummulative_reward/(e+1)
            cummulative_score += score
            average_score = cummulative_score/(e+1)
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, score, agent.epsilon))
            result_writer.writerow({'episode':e, 'epsilon':agent.epsilon, 'score':score, 'average_score':average_score,
                                    'total_reward': total_reward, 'average_reward':average_reward})
            break

    if len(agent.memory) > batch_size:
        # train the agent with the experience of the episode
        agent.replay(batch_size)

# testing trials
print("///////////TESTING/////////")
agent.epsilon = 0.0
cummulative_score = 0
cummulative_reward = 0
test_results = open('test_results_DQN_model1.csv', 'w',newline='')
result_writer = csv.DictWriter(test_results,fieldnames)
result_writer.writeheader()
for e in range(50):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    for time in range(500):
        # for GUI
        env.render()

        # Decide action
        action = agent.act(state)

        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        total_reward += reward
        next_state = np.reshape(next_state, [1, state_size])

        # make next_state the new current state for the next frame.
        state = next_state

        # done becomes True when the game ends
        # ex) The agent drops the pole
        if done:
            # print and store the metrics and break out of the loop
            cummulative_reward += total_reward
            average_reward = cummulative_reward / (e+1)
            cummulative_score += score
            average_score = cummulative_score / (e+1)
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e+1, 50, score, agent.epsilon))
            result_writer.writerow({'episode':e+1, 'epsilon':agent.epsilon, 'score':score, 'average_score':average_score,
                                    'total_reward': total_reward, 'average_reward':average_reward})
            break