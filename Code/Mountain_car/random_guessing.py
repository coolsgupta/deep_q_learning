import gym
import random
import csv
import numpy as np

env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
fieldnames = ['episode', 'epsilon', 'score', 'average_score', 'total_reward', 'average_reward']
result_random_guessing = open('test_results_random_guessing.csv','w',newline='')
writer = csv.DictWriter(result_random_guessing,fieldnames)
writer.writeheader()
cumulative_reward = 0
cumulative_score = 0
for e in range(50):
    state = env.reset()
    state = np.reshape(state, [1,state_size])
    total_reward = 0
    score = state[0][0]
    for time in range(200):
        action = random.randrange(env.action_space.n)
        next_state,reward,done,_ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state,[1,state_size])
        total_reward += reward
        state = next_state
        score = state[0][0] if state[0][0] > score else score
        if done:
            cumulative_reward += total_reward
            average_reward = cumulative_reward / (e + 1)
            cumulative_score += score
            average_score = cumulative_score / (e + 1)
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e+1, '50', score, '1.0'))
            #result = ('Episode :', str(e), ' score:', str(score), ' epsilon:', '1.0', '\n')
            writer.writerow(
                {'episode': e, 'epsilon': '1.0', 'score': score, 'average_score': average_score,
                 'total_reward': total_reward, 'average_reward': average_reward}
            )
            break