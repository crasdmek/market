# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:53:00 2017

@author: jeremyfix
"""


import numpy as np
import random
import gym
from gym import spaces
import pandas as pd
import datetime as dt

class Market(gym.Env):
    pass

    def __init__(self, initial_cash = 10000, 
                 commission = 9.99, 
                 win_loss_pct = .12,
                 daily_loss_pct = .05/252):
        
        # Data Initialization Attributes
        self.initial_cash = initial_cash
        self.commission = commission
        self.win_loss_pct = win_loss_pct
        self.daily_loss_pct = daily_loss_pct
        
        self.observation_space = spaces.Discrete(64)
        
        # Actions
        self.actions = ["LONG", "SHORT", "NEUTRAL"]
        
        # Primary Class Attributes
        self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = []
        
        self.set_data()
        self._reset()

        
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
        info = {}
        
        # Set Negative Reward by Daily Loss Percentage * Days since start (data_index + 1)
        reward = -1 * self.daily_loss_pct * (self.data_index + 1)
        
        # Initialize observations as empty array
        observations = np.empty(0)
        
        # Increment Data Index
        self.data_index += 1  
                
        if self.data_index <= len(self.data) - 1:
            close = float(self.data['Close'][self.data_index])
            #yesterday_close = float(self.data['Close'][self.data_index-1])
            win_loss_pct = (self.net_value(close) - self.initial_cash)/self.initial_cash
                
            # If Gain is Greater Than Gain Target (End and Report Reward)
            if win_loss_pct >= self.win_loss_pct:
                self.done = True
                reward = win_loss_pct
                
            # If Gain is Less Than Loss Target (End and Report Reward)
            elif win_loss_pct <= -1 * self.win_loss_pct:
                self.done = True
                reward = win_loss_pct
                
            # Handle Action
            elif self.actions[action] == 'LONG':
                # IF SHORT, Buy back Contracts
                if self.contracts <= 0:
                    self.cash += self.contracts * close
                # Purchase as Many as allowed minus commission
                    self.contracts = int((self.cash - self.commission)/close)
                    self.cash -= self.contracts * close - self.commission

            elif self.actions[action] == 'SHORT':
                # IF LONG, Sell Contracts
                if self.contracts >= 0:
                    self.cash += self.contracts * close
                # Sell as Many as allowed minus commission
                    self.contracts = -1 * int((self.cash - self.commission)/close)
                    self.cash -= self.contracts * close - self.commission

            elif self.actions[action] == 'NEUTRAL':
                # If LONG, Sell Contracts
                if self.contracts > 0:
                    self.cash += self.contracts * close - self.commission
                    self.contracts = 0
                # If Short, Buy back Contracts
                elif self.contracts < 0:
                    self.cash += self.contracts * close - self.commission
                    self.contracts = 0
                else:
                    pass
            else:
                pass
            
            #observation = self.get_observation(close, yesterday_close, verbose = 0)
            # Observe last 4 days
            for i in range(4):
                today = float(self.data['Close'][self.data_index-i])
                yesterday = float(self.data['Close'][self.data_index-i-1])
                observation = self.get_observation(today, yesterday, verbose = 0)
                observations = np.append(observations, observation)
            
            observations = np.array(observations)
            
            info = {'Steps': self.data_index}
        else:
            self.done = True
            info = {'Steps': self.data_index}
            observations = np.ones(64)
                                  
        return observations, reward, self.done, info
    
    def _reset(self):
        """Resets the state of the environment and returns an initial observation.
        
        Returns: observation (object): the initial observation of the
            space.
        """
        
        #i = random.random()
        self.done = False
        self.data_index = 0
        self.cash = self.initial_cash
        self.contracts = 0
        self.daily_returns = np.array([])
        observation = np.ones(64)
        numdays = 500
        start = dt.date(1990, 2, 16)
        #end = dt.date(2016, 6, 6)
        #self.start_date = self.random_date(start, end)
        self.start_date = start
        end_date = self.start_date + dt.timedelta(days=numdays)
        self.data = self.master_data.loc[self.start_date:end_date]
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

        #i = random.random()
        observation = np.ones(16)        
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
            
            observation = np.array([daily_return, 
                                    pct_contracts, 
                                    pct_cash, 
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
                                    

        
        return observation
        
    def net_value(self, close):
        return self.contracts * close + self.cash
        
    def random_date(self, start, end):
        """Generate a random datetime between `start` and `end`"""
        return start + dt.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())),
        )
        
    def set_data(self):
        data = pd.read_csv('output.csv', index_col = 'DATE', parse_dates=True)
        data=data.rename(columns = {'SPX':'Close'})
        data=data.fillna(method='bfill')
        self.master_data = data
        
        

        

        

        
    





    

    
    