from ..simulator import simulator
from ..agent import Agent

class agent(Agent):
    def __init__(self, model, log_metrics = True, max_episodes = 1000, epsilon = 1, alpha = 1 ):
        Agent.__init__(self, model = model, epsilon = epsilon, alpha = alpha)
        self.name = "CartPole-v1"
        algo = 'table_based_q_learning' if model == 0 else 'Deep_Q_learning'
        self.sim = simulator(agent = self, env = self.name, algo = algo, log_metrics= log_metrics, max_episodes= max_episodes)
