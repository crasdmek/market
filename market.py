# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:53:00 2017

@author: jeremyfix
"""


#import datetime
import numpy as np
import random
#import math
import gym
from gym import spaces
import sys

class Market(gym.Env):
    pass

    def __init__(self, initial_cash = 10000, commission = 9.99, extended = False):
        
        # Data Initialization Attributes
        self.initial_cash = initial_cash
        self.commission = commission
        self.extended = extended
        
        if self.extended:
            self.observation_space = spaces.Box(np.ones(16), np.ones(16))
        else:
            self.observation_space = spaces.Box(np.ones(3), np.ones(3))
        
        # Actions
        self.actions = ["BUY", "SELL", "HOLD"]
        
        # Primary Class Attributes
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = []
        
        self._reset()
        #self._seed()
        
    def _step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        
        Accepts an action and returns a tuple (observation, reward, done, info).
        
        Args:
            action (object): an action provided by the environment
            
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        observation = self.get_observation()
        info = {}
        reward = 0.0000000001
        margin_limit = 1.5
        short_factor = -.5
        
        self.data_index += 1    
        if self.data_index <= len(self.data) - 1:
            close = float(self.data['Close'][self.data_index])
            yesterday_close = float(self.data['Close'][self.data_index-1])
                        
            # Handle Action
            if self.actions[action] == 'BUY':
                if self.cash * margin_limit >= close + self.commission:
                    self.contracts += 1
                    self.cash = self.cash - close - self.commission
                    self.buys += 1
            elif self.actions[action] == 'SELL':
                if self.cash >= (self.contracts - 1) * close * short_factor:
                    self.contracts -= 1
                    self.cash = self.cash + close - self.commission
                    self.sells += 1
            elif self.actions[action] == 'HOLD':
                self.holds += 1
            else:
                pass
            
            observation = self.get_observation(close, yesterday_close, verbose = 0)
            
            info = {'Date': self.data.index.values[self.data_index],
                    'Action': self.actions[action],
                    'Close': close,
                    'Net_Value': self.net_value(close),
                    'Cash': self.cash,                      # validated
                    'Contracts': self.contracts
                    }
        else:
            self.done = True
            reward = self.get_sharpe_ratio()
            info = {'Buys:': self.buys,
                          'Sells:': self.sells,
                          'Holds:': self.holds,
                          'Reward:': reward}
                          
            
            verbose = 0
            if verbose == 1:
                print info
        
        return observation, reward, self.done, info
    
    def _reset(self):
        """Resets the state of the environment and returns an initial observation.
        
        Returns: observation (object): the initial observation of the
            space.
        """
        
        i = random.random()
        self.done = False
        self.data_index = 0
        self.cash = self.initial_cash
        self.contracts = 0
        self.buys = 0
        self.sells = 0
        self.holds = 0
        self.daily_returns = np.array([])
        if self.extended:
            observation = np.array([i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i])
        else:
            observation = np.array([i,i,i])
        return observation

    
    def _render(self, mode='human', close=False):
        pass
    
    def _close(self):
        pass
    
    def _configure(self):
        pass
    
    def _seed(self):
        return int(random() * 100)

    
    def get_observation(self, close = 0.00, yesterday_close = 0.00, verbose = 0):

        i = random.random()
        if self.extended:
            observation = np.array([i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i])
        else:
            observation = np.array([i,i,i])
        
        if self.done == False:
            gain_loss = (close - yesterday_close) * self.contracts
            if yesterday_close > 0:
                daily_return = gain_loss/self.net_value(yesterday_close)
            else:
                daily_return = 0.0
            contracts_value = self.contracts * close
            net_value = self.net_value(close)
            pct_contracts =  contracts_value / net_value
            pct_cash = self.cash / net_value
            
            self.daily_returns = np.append(self.daily_returns, daily_return)
            #self.daily_returns.append(daily_return)
            
            if self.extended:
                observation = np.array([daily_return, pct_contracts, pct_cash, 
                                        self.data['VIX'][self.data_index],
                                        self.data['High'][self.data_index],
                                        self.data['Low'][self.data_index],
                                        self.data['Vol/AvgVol'][self.data_index],
                                        self.data['Adv/Dec'][self.data_index],
                                        self.data['AVol/DVol'][self.data_index],
                                        self.data['RSI'][self.data_index],
                                        self.data['ROC'][self.data_index],
                                        self.data['MACD'][self.data_index],
                                        self.data['ATR'][self.data_index],
                                        self.data['ADOSC'][self.data_index],
                                        self.data['DayOfWeek'][self.data_index],
                                        self.data['Month'][self.data_index]])
            else:
                observation = np.array([daily_return, pct_contracts, pct_cash])
        
        if verbose == 1:
            print observation
        return observation
        
    def net_value(self, close):
        return self.contracts * close + self.cash
        
    def set_data(self, data):
        self.data = data
        
    def get_sharpe_ratio(self):        
        # Compute Average Daily Returns
        avg_daily_ret = self.daily_returns.mean()
        
        # Compute Standard Deviation of Daily Returns
        std_daily_ret = self.daily_returns.std()
        
        # Compute Portfolio Sharpe Ratio
        k = np.sqrt(len(self.daily_returns))
        sharpe_ratio = k * avg_daily_ret/std_daily_ret
        if std_daily_ret == 0:
            return 0.0000000001
        else:
            return sharpe_ratio
        

        
    





    

    
    