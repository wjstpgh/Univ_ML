# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 23:57:13 2021

@author: tmark
"""
#회귀분석을 위한 세팅
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#회귀분석중 단순선형방식,테스트방식,테스트데이터분리
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#단순선형회귀분석
house=pd.read_csv('C:\BostonHousing.csv')
house.columns

lstat=np.array(house['lstat']).reshape(506,1)
medv=np.array(house['medv']).reshape(506,1)
model=LinearRegression()
model.fit(lstat,medv)
pred_y=model.predict(lstat)

print(pred_y)
lstat
medv
 
#모델 회귀식 
print('medv = {0}(W) x lstat + {1}(b)'\
      .format(model.coef_[0][0], model.intercept_[0]))

#회귀식이용 medv예측
w=model.coef_[0][0]
b=model.intercept_[0]

for x in range (2,6):
    print('lstat의 값이 ' + str(x) + '일때 medv의 예측값은:')
    print('회귀식예측:{0}'.format(w*x+b))
    print('모델예측:{0}'.format(model.predict([[x]])))

#MSE 계산
pred_y=model.predict(lstat)
print('Mean squared error:{0}'.format(mean_squared_error(medv,pred_y)))

#다중선형회귀분석
medv_x=house[['lstat','ptratio','tax','rad']]
medv_y=house['medv']
model=LinearRegression()
model.fit(medv_x,medv_y)
pred_y=model.predict(medv_x)
print(pred_y)

#모델 회귀식
print('medv = {0}xlstat + {1}xptratio + {2}xtax + {3}xrad + {4}'.format(model.coef_[0], model.coef_[1],model.coef_[2],model.coef_[3],model.intercept_))

#다중모델예측값
test=np.array([[2.0,14,296,1],[3.0,15,222,2],[4.0,15,250,3]]).reshape(3,-1)
test_pred_y=model.predict(test)
print(test_pred_y)

#다중모델MSE계산
pred_y=model.predict(medv_x)
print('Mean squared error:{0}'.format(mean_squared_error(medv_y,pred_y)))

#논리회귀분석
ucla=pd.read_csv('C:\\ucla.csv')
ucla

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ucla_x=ucla[['gre','gpa','rank']]
ucla_y=ucla['admit']
model=LogisticRegression()

train_x,test_x,train_y,test_y=train_test_split(ucla_x,ucla_y,test_size=0.3,random_state=1234)
model.fit(train_x,train_y)
pred_y=model.predict(test_x)
print(pred_y)

#training, test accuracy
pred_train_y=model.predict(train_x)
train_acc=accuracy_score(train_y, pred_train_y)
test_acc=accuracy_score(test_y, pred_y)
print('training accuracy:{0}, test accuracy:{1}'.format(train_acc, test_acc))

#논리모델예측값
test=np.array([[400,3.5,5],[550,3.8,2],[700,4.0,2]]).reshape(3,-1)
test_pred_y=model.predict(test)
print(test_pred_y)

#gre, gpa두 변수로만예측
ucla_x=ucla[['gre','gpa']]

train_x,test_x,train_y,test_y=train_test_split(ucla_x,ucla_y,test_size=0.3,random_state=1234)
model.fit(train_x,train_y)
pred_y=model.predict(test_x)
print(pred_y)

#정확도 측정
pred_train_y=model.predict(train_x)
train_acc=accuracy_score(train_y, pred_train_y)
test_acc=accuracy_score(test_y, pred_y)
print('training accuracy:{0}, test accuracy:{1}'.format(train_acc, test_acc))











