# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def choose_multinomial(probs):
    r = random.random()
    index = 0
    while(r >= 0 and index < len(probs)):
        r -= probs[index]
        index += 1
    return index - 1


# Creating the architecture of the Neural Network

class Network:
    
    def __init__(self, input_size, nb_action):
        self.input_size = input_size
        self.nb_action = nb_action #output size (number of possible actions)
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.file_name = "car_keras.h5"


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(30, input_dim=self.input_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.nb_action, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def load(self):
        self.model.load_weights(self.file_name)

    def save(self):
        self.model.save_weights(self.file_name)

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        return map(list, zip(*random.sample(self.memory, batch_size)))

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.last_state = np.zeros((1,input_size))
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = softmax(self.model.model.predict(state)[0] *100)
        return choose_multinomial( probs) #next is chosen random by prediction percentages

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        batch_state = np.array(batch_state)
        outputs = self.model.model.predict(batch_state)
        next_outputs = np.amax(self.model.model.predict(np.array(batch_next_state)), axis=1)
        target = self.gamma*next_outputs + batch_reward

        for idx, output in enumerate(outputs):
            output[batch_action[idx]] = target[idx]

        self.model.model.fit(batch_state, outputs, epochs=1, verbose=0)
    
    def update(self, reward, new_signal):
        new_state = np.array(new_signal).reshape(1,len(new_signal))
        self.memory.push((self.last_state[0], new_state[0], self.last_action, self.last_reward))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        self.model.save()
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            self.model.load()
            print("done !")
        else:
            print("no checkpoint found...")