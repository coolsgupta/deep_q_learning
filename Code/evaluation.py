import matplotlib.pyplot as plt

class evaluate(object):
    def average_score_per_episode(self, scores, episodes):
        plt.xlabel('Epsiode number')
        plt.ylabel('Average score per episode')
        average_scores = []
        for x in range(episodes):
            average_scores.append(scores[x]/x)
        plt.plot((x for x in range(episodes)), average_scores)


    def total_score(self, scores, episodes):
        plt.xlabel('Episode')
        plt.ylabel('total score')
        plt.plot((x for x in range(episodes)), scores)