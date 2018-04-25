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
        elif model == 2:
            self.DQN_CNN()
        else:
            print ("Invalid model number")

    def Q_learning_model(self):
        return

    def DQN_DNN(self):
        model = Sequential()
        model.add(Dense(32, input_shape=state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.summary()
        return

    def DQN_CNN(self):
        return