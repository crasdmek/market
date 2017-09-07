# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 20:17:07 2017

@author: jeremyfix
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import numpy as np

class Model():
    def __init__(self, input_dim=1, output_dim=1):
        self.build_model(input_dim, output_dim)
    
    def train(self, x, y, verbose=1):
        self.model.fit(x, y, verbose, epochs=1)
    
    def predict(self, x, verbose=False):
        x = np.array(x)
        if verbose:
            print x
            print x.shape
        y = self.model.predict(x)
        return y
        
    def build_model(self, input_dim, output_dim, verbose=True):
        # Setup Keras Sequential Model
        self.model = Sequential()
        
        # Input Layer
        self.model.add(Dense(units=64, input_dim=input_dim))
        self.model.add(Activation('relu'))
        
        self.model.add(Dense(units=32))
        self.model.add(Activation('relu'))
        
        self.model.add(Dense(units=16))
        self.model.add(Activation('relu'))
        
        # Output Layer
        self.model.add(Dense(units=output_dim))
        self.model.add(Activation('softmax'))
        
        # Configure Learning
        self.model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
              
        if verbose: 
            self.model.summary()
            print "Input: " + str(input_dim)
            print "Output: " + str(output_dim)
