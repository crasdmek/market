

import random as rand
from learner import QLearner
from market_env import Market
import numpy as np
    
# run the code to test a learner
if __name__=="__main__":

    verbose = False #print lots of debug stuff if True
            
    rand.seed(5)

    ''' Initialize Market Environment '''
    env = Market(initial_cash = 10000, 
             commission = 9.99, 
             win_loss_pct = .12,
             daily_loss_pct = .24/252)
             
    ''' Retrieve State and Action Space from Environment '''
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    ''' Initialize Learning Agent '''
    agent = QLearner(\
        num_states,\
        num_actions, \
        alpha = 0.2, \
        gamma = 0.98, \
        rar = 0.98, \
        radr = 0.999, \
        dyna = 0, \
        verbose=False)
    
        # GAMMA = discount factor, higher tends toward longer term rewards
        # ALPHA = learning rate, higher tends toward deterministic
        
    batch_cnt = 100
        
    for batch in range(batch_cnt):
            
        ''' Batch Variables for Training an Entire Batch at Once '''
        #S = []
        #Q = []
        
        # Counter for Wins and Losses
        wins = 0
        losses = 0
        
        ''' Iterate through X Episodes to Train Learner '''
        for episode in range(100):  
            done = False
            s = env.reset()
            a = agent.querysetstate(s)
            cumulative_reward = 0
            while not done:
                s_prime, r, done, info = env.step(a)
                #cumulative_reward += r
                a = agent.get_action(s, s_prime, r)
                s = s_prime
                
            #print info
                
            ''' Retrieve Tables from Agent '''
            # s, q = agent.get_q()
            
            ''' Add New Data to S, Q '''
            # S = S + s
            # Q = Q + q
            
            ''' Reset Agent '''
            agent.reset()
            
            print "Batch: " + str(batch) + "\tEpisode: " + str(episode) + "\tResult = " + str(info['Result']) + "\tReturn: " + str(info['Return'])
            if r > 0: 
                wins += 1
            elif r < 0:
                losses += 1
        
        print "Batch: " + str(batch) + "\tWins: " + str(wins) + "\tLosses: " + str(losses) + "\tRAR: " + str(agent.rar)
            
                    
        ''' Convert to Numpy Arrays '''
        #S = np.array(S)
        #Q = np.array(Q)
            
        ''' Train Models from Tables '''
        # agent.train(S, Q)
        
        ''' Slowly Decrease RAR (Random Action Rate) depending on # of Batches '''
        agent.decrease_rar(1.0/batch_cnt)

   
