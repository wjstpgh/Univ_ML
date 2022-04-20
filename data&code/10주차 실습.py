# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:02:17 2021

@author: tmark
"""

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sys,os
sys.path.append(os.path.abspath('D:\\'))
import trainplot

df=pandas.read_csv('C:\PimaIndiansDiabetes.csv')
df=df.values
df_x=df[:,0:-1].astype(float)
df_y=df[:,-1]

enc=LabelEncoder() 
enc.fit(df_y)
df_y=enc.transform(df_y)
df_y=np_utils.to_categorical(df_y)
#df_y pos=[0,1],1 neg=[1,0],0

tr_x,te_x,tr_y,te_y=train_test_split(df_x,df_y,test_size=0.3,random_state=123)

epochs=80
batch_size=30
learn_rate=0.01

model=Sequential()
model.add(Dense(160,input_dim=8,activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(170,activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(180,activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(180,activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(2,activation='softmax'))

model.summary()
adam=optimizers.adam_v2.Adam(learning_rate=learn_rate)
dir(optimizers)

model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
disp=model.fit(tr_x,tr_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.1,callbacks=[trainplot.TrainingPlot()])





















