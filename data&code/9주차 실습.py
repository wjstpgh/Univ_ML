# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:17:58 2021

@author: tmark
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas
import matplotlib.pyplot as plt
import numpy as np

df=pandas.read_csv('C:/liver.csv')
ds=df.values
x=ds[:,1:].astype(float)
y=ds[:,0]
y=np_utils.to_categorical(y)

tr_x,te_x,tr_y,te_y=train_test_split(x,y,test_size=0.4,random_state=123)

epochs=500
batch_size=10

model=Sequential()
model.add(Dense(10,input_dim=6,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

disp=model.fit(tr_x,tr_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(te_x,te_y))
pred=model.predict(te_x)
pred_y=[np.argmax(y,axis=None,out=None)for y in pred]

score=model.evaluate(te_x,te_y,verbose=0)
print('loss:',score[0],'acc:',score[1])

#히든레이어 한층 추가
model=Sequential()
model.add(Dense(10,input_dim=6,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

disp=model.fit(tr_x,tr_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(te_x,te_y))
pred=model.predict(te_x)
pred_y=[np.argmax(y,axis=None,out=None)for y in pred]

score=model.evaluate(te_x,te_y,verbose=0)
print('epochs: ',epochs,'일때, loss:',score[0],'acc:',score[1])

#그래프
plt.plot(disp.history['accuracy'])
plt.plot(disp.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()











































