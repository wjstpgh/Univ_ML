# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:49:09 2021

@author: tmark
"""
import numpy as np

def SIGMOID(x):
    return 1/(1+np.exp(-x))

#Delta rule
x=np.array([0.2,0.9,0.5])
w=np.array([0.3,0.4,0.5])
d=1
alp=0.5

for D in range(10):
    y=np.sum(x*w)
    e=d-y
    print('error',D,e,sep=':')
    w=w+alp*e*x


#simple Delta rule
x=np.array([0.5,0.8,0.2])
w=np.array([0.4,0.7,0.8])
d=1
alp=0.5

for t in range(50):
    v=np.sum(x*w)
    y=SIGMOID(v)
    e=d-y
    print('error',t,e)
    w=w+alp*y*(1-y)*e*x
    #for a in range(3):
       #w[a]=w[a]+alp*y*(1-y)*e*x[a]













































