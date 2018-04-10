from .models import models

class Agent(object):
    def __init__(self, epsilon, alpha, model = 0):
        '''
        :param model: 0 = table based q learning model
                      1 = deep q learning - Deep neural network model
                      2 = deep q learning - Convolutional neural network model
        '''
        self.model = models(model)

    def train(self):


    def train_with_episode_experience(self):

