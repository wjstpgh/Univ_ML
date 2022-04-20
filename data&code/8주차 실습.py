# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:57:49 2021

@author: tmark
"""

from sklearn import datasets,model_selection
import random
import numpy as np

iris=datasets.load_iris()
x=iris.data 
target=iris.target

x_tr,x_te,y_tr,y_te=model_selection.train_test_split(x,target,test_size=0.3,random_state=123)

num=np.unique(y_tr,axis=0)
num=num.shape[0]
y=np.eye(num)[target]

n=x.shape[1]*y.shape[1]
random.seed=123
w=random.sample(range(1,100),n)
w=(np.array(w)-50)/100
w=w.reshape(x.shape[1],-1)

def SIGMOID(x):
    return 1/(1+np.exp(-x))

pred=np.zeros(x_tr.shape[0])
for i in range(x_tr.shape[0]):
    v=np.matmul(x_tr[i,:],w)
    y1=SIGMOID(v)
    pred[i]=np.argmax(y1)
    print('target, predict', y_tr[i],pred[i])
    
print('tr_acc:',np.mean(pred==target))

pred=np.zeros(x_te.shape[0])
for i in range(x_te.shape[0]):
    v=np.matmul(x_te[i,:],w)
    y1=SIGMOID(v)
    pred[i]=np.argmax(y1)
    print('target, predict', y_te[i],pred[i])
    
print('te_acc:',np.mean(pred==target))

def SLP_SGD(tr_x,tr_y,w,alpha,rep,batch):
    w1=w
    b=tr_x.shape[0]//batch
    a=np.zeros(shape=(4,3), dtype=float)
    count=1
    for i in range(rep):
        for k in range(tr_x.shape[0]):
            x=tr_x[k,:]
            v=np.matmul(x,w1)
            y=SIGMOID(v)
            e=tr_y[k,:]-y
            a+=w1+(x.reshape(4,1)*(alpha*y*(1-y)*e))
            count+=1
            if count==b:
                w1=a/b
                count=1
                a=np.zeros(shape=(4,3), dtype=float)
        w1=a/count    
        a=np.zeros(shape=(4,3), dtype=float)
        print('error',i,np.mean(e))
    return w1

w=SLP_SGD(x_tr,y,w,alpha=0.01,rep=50,batch=10)



