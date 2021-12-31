import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import rl
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, LSTMCell, GRU
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent, DDPGAgent, CEMAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import scipy.signal
from gym import Env
from gym.spaces import Discrete, Box
import random
from EXPE import StockTradingEnv_EXPE
from _APA import StockTradingEnv_APA

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(10, 6,6)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(actions, activation='softmax'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=10)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn
#EXPE
env_EXPE = StockTradingEnv_EXPE()
n_states_EXPE = env_EXPE.observation_space.shape
n_actions_EXPE= env_EXPE.action_space.n
model_EXPE = build_model(n_states_EXPE, n_actions_EXPE)
model_EXPE.summary()
dqn_EXPE = build_agent(model_EXPE, n_actions_EXPE)
dqn_EXPE.compile(Adam(learning_rate=0.0001), metrics= ['mae'])
dqn_EXPE.fit(env_EXPE, nb_steps=env_EXPE.steps, visualize=True, verbose=0.1)

#APA
env_APA = StockTradingEnv_APA()
n_states_APA = env_APA.observation_space.shape
n_actions_APA= env_APA.action_space.n
model_APA = build_model(n_states_APA, n_actions_APA)
model_APA.summary()
dqn_APA = build_agent(model_APA, n_actions_APA)
dqn_APA.compile(Adam(learning_rate=0.0001), metrics= ['mae'])
dqn_APA.fit(env_APA, nb_steps=env_APA.steps, visualize=True, verbose=0.1)

#KMI
env_KMI = StockTradingEnv_KMI()
n_states_KMI = env_KMI.observation_space.shape
n_actions_KMI = env_KMI.action_space.n
model_KMI = build_model(n_states_KMI, n_actions_KMI)
model_KMI.summary()
dqn_KMI = build_agent(model_KMI, n_actions_KMI)
dqn_KMI.compile(Adam(learning_rate=0.0001), metrics= ['mae'])
dqn_KMI.fit(env_KMI, nb_steps=env_KMI.steps, visualize=True, verbose=0.1)