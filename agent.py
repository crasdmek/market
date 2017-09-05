"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
#import pandas as pd

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna        
        self.q = self.initQ(num_states, num_actions)
        self.R = self.initR(num_states, num_actions)
        self.Tc = self.initTc(num_states, num_actions)


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action
        

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The next state
        @returns: The selected action
        """
        action = 0
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.q[s_prime, :])
            
        self.updateQ(self.s, self.a, s_prime, r, action)
        self.updateR(self.s, self.a, r)
        self.updateTc(self.s, self.a, s_prime)
        
        for i in range(0, self.dyna):
            s = rand.randint(0, self.num_states-1)
            a = rand.randint(0, self.num_actions-1)
            s1 = np.argmax(self.Tc[s, a, :])
            r = self.R[s, a]
            a1 = 0
            if rand.random() < self.rar:
                a1 = rand.randint(0, self.num_actions-1)
            else:
                a1 = np.argmax(self.q[s1, :])
            self.updateQ(s, a, s1, r, a1)

        self.s = s_prime
        self.a = action
        
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r,"rar = ",self.rar
        return action
    
    def initQ(self, num_states, num_actions):
        return np.zeros((num_states, num_actions)) # Initial to all Zeros
        
    def initR(self, num_states, num_actions):
        return np.zeros((num_states, num_actions)) # Initial to all Zeros
        
    def initTc(self, num_states, num_actions):
        Tc = np.zeros((num_states, num_actions, num_states))
        Tc[:,:,:] = .00001
        return Tc
    
    def updateQ(self, s, a, s_prime, r, a_prime):
        self.q[s,a] = (1 - self.alpha) * self.q[s,a] + self.alpha * (r + self.gamma * self.q[s_prime, np.argmax(self.q[s_prime, a_prime])])
        self.rar = self.rar * self.radr # decay random action
        
    def updateR(self, s, a, r):
        self.R[s, a] = (1 - self.alpha) * self.R[s, a] + self.alpha * r
        
    def updateTc(self, s, a, s1):
        self.Tc[s, a, s1] += 1

            
if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
