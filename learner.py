import numpy as np
import random as rand
from model import Model

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.99, \
        radr = 1, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha #LEARNING RATE
        self.gamma = gamma #DISCOUNT FACTOR
        self.rar = rar #random action rate
        self.radr = radr
        self.dyna = dyna        
        self.q_model = Model(num_states, num_actions)
        self.reset()
        
    def querysetstate(self, s):
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action
        

    def get_action(self,s, s_prime, r):
        action = 0
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.q_model.predict([s_prime])[0])
        self.updateQ(s, action, s_prime, r)
        return action
        
    def updateQ(self, s, a, s_prime, r, verbose=False):
        #self.q[s,a] = (1 - self.alpha) * self.q[s,a] + self.alpha * (r + self.gamma * self.q[s_prime, np.argmax(self.q[s_prime, a_prime])])       
       
        q_sa = self.q_model.predict([s])[0][a]
        max_q_s_prime = np.argmax(self.q_model.predict([s_prime])[0])
        #q_s_prime = self.q_model.predict([s_prime])[0][max_q_s_prime]
        q = q_sa + self.alpha * (r + self.gamma * max_q_s_prime - q_sa)
        
        # DEBUG
        if verbose: print str(q) + "\r\r"     
        
        # Initialize Q
        Q = np.zeros(self.num_actions)
        
        # Populate Q with Current Values from Model
        for i in range(self.num_actions):
            Q[i-1] = self.q_model.predict([s])[0][i-1]
            
        # Replace Q for Given Action a with Updated Q Value q
        Q[a] = q
        
        # Append New Q Values to Q Lists        
        #self.q_states.append(s)
        #self.q_qs.append(Q)
        s = s.reshape(1,64)
        Q = Q.reshape(1,3)
        self.train(s, Q)
                
    def get_q(self):
        return self.q_states, self.q_qs
        
    def reset(self):
        self.q_states = []
        self.q_qs = []
        self.s = 0
        self.a = 0   
        
    def train(self, x, y):
        self.q_model.train(x, y)
        
    def decrease_rar(self, x):
        self.rar = self.rar - x
            
if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
