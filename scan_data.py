# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 04:28:01 2017

@author: jeremyfix
"""

import pandas as pd

df = pd.read_csv('history.csv')
df.plot()
print df