import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils as np

class models(object):
    def __init__(self, model):

        if model == 0:
            self.Q_learning_model()
        elif model == 1:
            self.DQN_DNN()
        else:
            self.DQN_CNN()

    def Q_learning_model(self):

    def DQN_DNN(self):

    def DQN_CNN(self):