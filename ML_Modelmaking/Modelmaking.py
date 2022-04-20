# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 00:42:18 2021

@author: tmark
"""
#분석하고자 하는 데이터 파악
from sklearn import model_selection
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint

df=pd.read_csv('C:/train_open.csv')
df
df.columns
data = df.values.tolist()
df=pd.DataFrame(data,columns=df.columns)

df_add=pd.DataFrame(list(df.columns))
df=df.append(df_add,ignore_index=True)
type(df.columns)
list(df.columns)
df_add

df.loc[(df['1.3']!=2)&(df['1.3']!=1)]

df_x=df.loc[:,df.columns!='1.3']
df_y=df['1.3']

#가우시안 정규분포를 위한 전처리
scaler=StandardScaler()
scaler.fit(df_x)#테스트데이터는 fit금지
scaled=scaler.transform(df_x)
df_x=pd.DataFrame(scaled, columns=df_x.columns)

#알고리즘 선택
model=[]
model.append(('LR',LogisticRegression()))
model.append(('RF',RandomForestClassifier()))
model.append(('KNN',KNeighborsClassifier()))
model.append(('LSVM',LinearSVC(max_iter=5000, C=100, dual=True)))
model.append(('SVM',SVC()))
model_result=[]
model_name=[]

kfold=model_selection.KFold(n_splits=10,random_state=123,shuffle=True)

for name,model in model:
    cv_result=model_selection.cross_val_score(model, df_x,df_y,cv=kfold,scoring='accuracy')
    model_name.append(name)
    model_result.append(cv_result)
    
print(model_result)

fig=plt.figure()
ax=fig.add_subplot(111)
plt.boxplot(model_result)
ax.set_xticklabels(model_name)
plt.show()

#속성선택
bestmodel=RandomForestClassifier(random_state=1234)
sfs1=SFS(bestmodel,k_features=10,verbose=2,scoring='accuracy',cv=kfold)
sfs1=sfs1.fit(df_x,df_y,custom_feature_names=df_x.columns)
sfs1.subsets_
sfs1.k_feature_idx_
sfs1.k_feature_names_ #전위선택결과

sel_fit=SelectFromModel(RandomForestClassifier())
sel_fit.fit(df_x,df_y)
sel_fit.get_support()
best_fit=df_x.columns[(sel_fit.get_support())]
len(best_fit)
best_fit #RF의 의사결정방식을 이용한 특징추출

sfs2=SFS(bestmodel,forward=False,k_features=12,verbose=2,scoring='accuracy',cv=kfold)
sfs2=sfs2.fit(df_x,df_y,custom_feature_names=df_x.columns)
sfs2.subsets_
sfs2.k_feature_idx_
sfs2.k_feature_names_ #후위버림결과 0.86

scores=cross_val_score(bestmodel,df_x[list(sfs1.k_feature_names_)],df_y,cv=kfold)
scores=cross_val_score(bestmodel,df_x[list(sfs2.k_feature_names_)],df_y,cv=kfold)
scores=cross_val_score(bestmodel,df_x[list(best_fit)],df_y,cv=kfold)
print(scores)
print(np.mean(scores))

df_x=df_x[list(sfs2.k_feature_names_)]

#하이퍼 파라미터 선택
pp=pprint.PrettyPrinter(width=80,indent=4)

param_grid={
    'n_estimators':[500,750],
    'max_depth':[13,15,17],
    'min_samples_split':[2,3],
    'min_samples_leaf':[1,2],
    'max_leaf_nodes':[None],
    #'min_impurity_decrease':[0.0,0.2],
    #'bootstrap':[True,False],
    #'ccp_alpha':[0.0,0.2,0.8],
    'oob_score':[True]
    }

grid_search=GridSearchCV(estimator=bestmodel, param_grid=param_grid, cv=5, n_jobs=-1,verbose=2)
grid_search.fit(df_x,df_y)
pp.pprint(grid_search.best_params_)

ran_search=RandomizedSearchCV(estimator=bestmodel, param_distributions=param_grid,n_iter=250,cv=5,verbose=2,random_state=123,n_jobs=-1)
ran_search.fit(df_x,df_y)
pp.pprint(ran_search.best_params_)

bestmodel=ran_search.best_estimator_

scores=cross_val_score(bestmodel,df_x,df_y,cv=kfold)
print(scores)
print(np.mean(scores))

#선택된 모델로 테스트데이터 예측하기
testdf=pd.read_csv('C:/test_open.csv')

bestmodel=RandomForestClassifier(n_estimators=500,max_depth=17,min_samples_split=2,min_samples_leaf=2,max_leaf_nodes=None,oob_score=True,random_state=1234)
bestmodel.fit(df_x,df_y)
testdf_columns=df.loc[:,df.columns!='1.3']

scaled=scaler.transform(testdf)
testdf=pd.DataFrame(scaled, columns=testdf_columns.columns)
testdf=testdf[list(sfs2.k_feature_names_)]
testdf

pred_y=bestmodel.predict(testdf)
pred_y=pd.DataFrame(pred_y)
pred_y
pred_y.columns=['answer']

pred_y.to_csv('D:/32144107_전세호.csv')


#반복생성
df=pd.read_csv('C:/train_open.csv')
df_x=df.loc[:,df.columns!='1.3']
df_y=df['1.3']

scaler=StandardScaler()
scaler.fit(df_x)#테스트데이터는 fit금지
scaled=scaler.transform(df_x)
df_x=pd.DataFrame(scaled, columns=df_x.columns)
df_x=df_x[list(sfs2.k_feature_names_)]
df_x=df_x[list(best_fit)]

bestmodel=RandomForestClassifier(n_estimators=500,max_depth=17,min_samples_split=2,min_samples_leaf=2,max_leaf_nodes=None,oob_score=True,random_state=1234)
bestmodel.fit(df_x,df_y)

testdf=pd.read_csv('C:/test_open.csv')
testdf_columns=df.loc[:,df.columns!='1.3']
scaled=scaler.transform(testdf)
testdf=pd.DataFrame(scaled, columns=testdf_columns.columns)
testdf=testdf[list(sfs2.k_feature_names_)]
testdf=testdf[list(best_fit)]

pred_y=bestmodel.predict(testdf)
pred_y=pd.DataFrame(pred_y)
pred_y.columns=['answer']
pred_y.to_csv('D:/32144107_전세호.csv')


