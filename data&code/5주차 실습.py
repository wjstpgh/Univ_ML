# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:18:55 2021

@author: tmark
"""

#의사결정트리 모델생성
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

df=pd.read_csv('C:\PimaIndiansDiabetes.csv')
df.head()
df.columns

df_x=df.loc[:,df.columns!='diabetes']
df_y=df['diabetes']

DT_model=DecisionTreeClassifier()
DT_score=cross_val_score(DT_model,df_x,df_y,cv=10)

print('의사결정트리의 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(DT_score, np.mean(DT_score)))

#랜덤포레스트 모델생성
from sklearn.ensemble import RandomForestClassifier

RF_model=RandomForestClassifier()
RF_score=cross_val_score(RF_model,df_x,df_y,cv=10)

print('Random Forest의 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(RF_score, np.mean(RF_score)))

#SVM 모델생성
from sklearn import svm

SVM_model=svm.SVC()
SVM_score=cross_val_score(SVM_model,df_x,df_y,cv=10)

print('Support Vector Machine의 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(SVM_score, np.mean(SVM_score)))

#SVM kernel인자 비교
SVM_model=svm.SVC(kernel=('linear'))
SVM_score=cross_val_score(SVM_model,df_x,df_y,cv=10)

print('linear의 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(SVM_score, np.mean(SVM_score)))

SVM_model=svm.SVC(kernel=('poly'))
SVM_score=cross_val_score(SVM_model,df_x,df_y,cv=10)

print('poly의 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(SVM_score, np.mean(SVM_score)))

SVM_model=svm.SVC()
SVM_score=cross_val_score(SVM_model,df_x,df_y,cv=10)

print('rbf의 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(SVM_score, np.mean(SVM_score)))

SVM_model=svm.SVC(kernel=('sigmoid'))
SVM_score=cross_val_score(SVM_model,df_x,df_y,cv=10)

print('sigmoid의 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(SVM_score, np.mean(SVM_score)))

#precomputed정방형행렬입력을 필요로함 어떻게 바꿔야할지모르겠음
np.shape(df_x)
drop_df_x=df_x.drop(df_x.index[0:46],axis=0)
np.shape(drop_df_x)
drop_df_y=df_y.drop(df_y.index[0:46],axis=0)
pre_df_x=np.reshape(drop_df_x,(76,-1))
df_x.isnull().sum()
SVM_model=svm.SVC(kernel=('precomputed'))
SVM_score=cross_val_score(SVM_model,drop_df_x,drop_df_y,cv=10)

print('precomputed의 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(SVM_score, np.mean(SVM_score)))

#RF인자 비교
for est in range(1,6):
    for mf in range(1,6):
        RF_model=RandomForestClassifier(n_estimators=100*est,max_features=mf,random_state=1234)
        RF_score=cross_val_score(RF_model,df_x,df_y,cv=10)
        
        print('({0},{1})일때 교차겁증값은 {2}이고, 평균값은 {3}입니다.'.format(est*100,mf,RF_score, np.mean(RF_score)))

#아래는 뻘짓
RF_model=RandomForestClassifier(n_estimators=100,max_features=1,random_state=1234)
RF_score=cross_val_score(RF_model,df_x,df_y,cv=10)

print('(100,1)일때 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(RF_score, np.mean(RF_score)))

RF_model=RandomForestClassifier(n_estimators=200,max_features=2,random_state=1234)
RF_score=cross_val_score(RF_model,df_x,df_y,cv=10)

print('(200,2)일때 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(RF_score, np.mean(RF_score)))

RF_model=RandomForestClassifier(n_estimators=300,max_features=3,random_state=1234)
RF_score=cross_val_score(RF_model,df_x,df_y,cv=10)

print('(300,3)일때 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(RF_score, np.mean(RF_score)))

RF_model=RandomForestClassifier(n_estimators=400,max_features=4,random_state=1234)
RF_score=cross_val_score(RF_model,df_x,df_y,cv=10)

print('(400,4)일때 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(RF_score, np.mean(RF_score)))

RF_model=RandomForestClassifier(n_estimators=500,max_features=5,random_state=1234)
RF_score=cross_val_score(RF_model,df_x,df_y,cv=10)

print('(500,5)일때 교차겁증값은 {0}이고, 평균값은 {1}입니다.'.format(RF_score, np.mean(RF_score)))






















