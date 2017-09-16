#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 10:59:23 2017

@author: jf9155
"""
from market_env import Market
import datetime as dt
import pylab
from market_dqn import DQNAgent
import numpy as np

if __name__ == "__main__":
    
    EPISODES = 20
    
    # MARKET_ENV VARIABLES
    INITIAL_CASH = 10000
    COMMISSION = 9.99
    
    # MARKET_DQN VARIABLES
    LOAD_MODEL = True
    EPSILON=0.00
    DISCOUNT_FACTOR = 0.99
    LEARNING_RATE = 0.001
    EPSILON_DECAY = 0.99999
    EPSILON_MIN = 0.00
    BATCH_SIZE = 64
    TRAIN_START = 1000
    MEMORY_SIZE = 4000

    scores, episodes = [], []

    for e in range(EPISODES):
        
        # INITIALIZE MARKET_ENV
        START_DATE = dt.date(1990+e, 2, 16)
        END_DATE = dt.date(1990+e, 4, 30)
        
        env = Market(initial_cash = INITIAL_CASH, 
                     commission = COMMISSION, 
                     start_date = START_DATE,
                     end_date = END_DATE)
                         
        # get size of state and action from environment
        STATE_SIZE = env.observation_space.n
        ACTION_SIZE = env.action_space.n
    
        # INITIALIZE DQNAgent
        agent = DQNAgent(state_size = STATE_SIZE, 
                         action_size = ACTION_SIZE,
                         load_model = LOAD_MODEL, 
                         discount_factor = DISCOUNT_FACTOR,
                         learning_rate = LEARNING_RATE, 
                         epsilon = EPSILON, 
                         epsilon_decay = EPSILON_DECAY, 
                         epsilon_min = EPSILON_MIN,
                         batch_size = BATCH_SIZE,
                         train_start = TRAIN_START, 
                         memory_size = MEMORY_SIZE)
    
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, STATE_SIZE])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, STATE_SIZE])

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/market_dqn.png")
                print("start_date:", START_DATE, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)


        # save the model
        if e % 50 == 0:
            agent.model.save_weights("./save_model/market_dqn.h5")
            
