import gym
import random
import csv

env = gym.make('CartPole-v1')
fieldnames = ['episode', 'epsilon', 'score', 'average_score', 'total_reward', 'average_reward']
result_random_guessing = open('test_results_random_guessing.csv','w',newline='')
writer = csv.DictWriter(result_random_guessing,fieldnames)
writer.writeheader()
cumulative_reward = 0
cumulative_score = 0
for e in range(50):
    state = env.reset()
    total_reward = 0
    for time in range(500):
        action = random.randrange(env.action_space.n)
        next_state,reward,done,_ = env.step(action)
        reward = reward if not done else -10
        total_reward += reward
        state = next_state
        if done:
            cumulative_reward += total_reward
            average_reward = cumulative_reward / (e + 1)
            cumulative_score += time
            average_score = cumulative_score / (e + 1)
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e+1, '50', time, '1.0'))
            result = ('Episode :', str(e), ' score:', str(time), ' epsilon:', '1.0', '\n')
            writer.writerow(
                {'episode': e, 'epsilon': '1.0', 'score': time, 'average_score': average_score,
                 'total_reward': total_reward, 'average_reward': average_reward}
            )
            break