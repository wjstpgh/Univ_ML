# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 00:10:24 2021

@author: tmark
"""

import numpy as np

def ptron(x):
    w=np.array([0.2,-0.1,0.3])
    
    v=np.sum(x*w)
    y=v if v>0 else 0
    return y

data_x=np.array([[0.3,0.1,0.8],
                 [0.5,0.6,0.3],
                 [0.1,0.2,0.1],
                 [0.8,0.7,0.7],
                 [0.5,0.5,0.6]])

for i in range(len(data_x)):
    print(ptron(data_x[i]))
    

