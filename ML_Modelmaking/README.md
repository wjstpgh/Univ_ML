# 머신러닝 모델 만들기

분석할 데이터 셋에 맞는 훈련모델을 제작하고 테스트한다.

![image](https://user-images.githubusercontent.com/26988563/164879154-41f56564-80ff-44fa-b75e-00d7e106e7e3.png)

우선 분석할 데이터를 파악한다. 데이터는 22개의 속성을 가지고 있고, 3999개의 데이터양을 보유함을 알 수 있다. 컬럼이름은 숫자로 되어있어 어떤 속성인지, 어떤 데이터를 분석한 것인지는 파악이 불가능하다. 

![image](https://user-images.githubusercontent.com/26988563/164879165-de537057-ed70-4d3e-b99e-0a2c391c87fd.png)

마지막 컬럼의 값이 1과 2만 존재하는지를 확인해보았다. 결과에 따라 이 데이터셋은 22개의 속성과 3999의 데이터양을 보유했고, 두가지로 분류하는 문제임을 알 수 있었다.

![image](https://user-images.githubusercontent.com/26988563/164879173-a05f15c2-dc4c-4037-968b-56d8495f7a6a.png)

우선 알고리즘을 선택하기 위해 싸이킷럿의 알고리즘 치트시트를 참고했다. 데이터셋의 성질에 따라 적용해볼만한 알고리즘들은 선형서포트머신, 앙상블모델, KNN등으로 추려졌다.

![image](https://user-images.githubusercontent.com/26988563/164879176-43d5c22a-958f-4000-bbfc-568a4b3c49d7.png)

로지스틱회귀, 랜덤포레스트, KNN, 선형서포트벡터머신 총 네 개를 실험해보았다. 오류가 떳는데 선형SVM이 수렴하지 않았다. 예상되는 원인에 따라 데이터셋의 문제로 쉽게 수렴할 수 있게 만들어 주거나, 파라미터 값을 조정하여 수렴계수를 늘리는 등의 방법을 알아보았다.

![image](https://user-images.githubusercontent.com/26988563/164879181-b771c929-6d4e-4783-b311-70044a081b35.png)

서포트 벡터머신이나 선형회귀는 애초에 가우시안 분포를 위해 만들어진 모델이기도 하고 다른 모델의 경우에도 분포도를 조정해주는 데이터 전처리를 하면 좀 더 예측모델의 정확도를 올릴 수 있을거란 판단으로 데이터를 가우시안정규분포를 위한 전처리를 진행했다.

![image](https://user-images.githubusercontent.com/26988563/164879189-b4c771cf-29f4-4b87-babe-95be86beb8fc.png)

전처리를 하고 작동시키니 ‘모델 피팅이 실패했고 정확도를 매길수 없다’는 오류가 떳다. 계속해서 이유를 찾던 중 y데이터를 보니 전처리 전의 1과2로 나눠졌던 데이터들이 전처리 후에는 완전히 연관성 없게끔 바뀐 것을 확인할 수 있었다. 나중에 안 사실이지만 전처리는 x데이터만 하는 것을 알게되었고, 이때 당시에는 일단 x만 전처리를 해봐야겠다는 생각으로 수정을 했다.

![image](https://user-images.githubusercontent.com/26988563/164879199-84f21b6b-879b-4ddd-b4f6-e01a2237cd31.png)

x에만 전처리를 시도했고 또다시 오류가 떳지만 마지막 컬럼값적용코드에서 x전처리 데이터에 전체 컬럼을 넣으려 해서 인덱스값 오류가 뜬 것이라 간단히 해결했다.

![image](https://user-images.githubusercontent.com/26988563/164879205-3633f48b-7d06-40bf-bd2c-4ef08ebea9f8.png)

xy를 나눠주고 그 후에 x데이터에만 가우시안 전처리를 다시 적용한 결과이다

![image](https://user-images.githubusercontent.com/26988563/164879215-da791ed1-aea6-42fa-980e-d18ab4619d3d.png)

전처리를 하고 선형 서포트벡터머신의 파라미터 값을 조정했는데도 불구하고 여전히 오류가 떴다. 그래서 Linear SVC은 무시하고 결과를 보기로 했다. 로지스틱회귀, 랜덤포레스트, KNN, 선형서포트벡터머신에서 Linear SVC이 작동안하는 대신 SVC를 추가했다.

![image](https://user-images.githubusercontent.com/26988563/164879224-7f84242e-98de-4688-a803-47b757b8e43b.png)

accuracy에서도 확인가능했고 그래프를 봐도 알 수 있듯이 랜덤포레스트가 압도적으로 좋은 성능을 나타내는 것을 확인 가능했다. 여기서 SVM과 병행해서 나머지 튜닝을 진행한다면 SVM도 좋은 효과를 낼 수도 있지만 차이도 크고 시간적 문제상 랜덤포레스트에 집중해서 튜닝을 하기로 결정했다.

![image](https://user-images.githubusercontent.com/26988563/164879231-2a298ba7-6d88-4689-9f8e-84851bf2d218.png)

모델은 랜덤포레스트로 픽스하고 이젠 속성선택을 했다. 전위선택방식으로 속성을 선택한 결과다.

![image](https://user-images.githubusercontent.com/26988563/164879237-807ca526-225b-4aee-9f2c-83ef07f90f44.png)

두 번째 속성선택방식은 후위제거방식을 사용해 5개의 속성을 남겨보았다.

![image](https://user-images.githubusercontent.com/26988563/164879243-f822cc52-3508-49a8-ae6a-4dbda1668ba5.png)

5개까지 제거를 했지만 정확도 변동을 보니 10개부터 급격히 줄어들어 속성값을 늘리는 것이 좋겠다는 판단을 했다.

![image](https://user-images.githubusercontent.com/26988563/164879253-dece73e8-53f6-4b8f-a977-5ec1904911c0.png)

마지막 방식은 DT의 의사결정방식에서 특징값들의 기여도를 계산하여 필요한 속성의 가중치를 계산하여 선택하는 방식인데 다른 두 개의 방식보다 좋은 효과를 보이지 않았다. 전위와 후위방식은 선택과정에서 변동하는 정확도 차이를 보았을 때 후위선택방식이 더 좋은 방식이라는 결론을 내리게 되었다.

![image](https://user-images.githubusercontent.com/26988563/164879259-c5e2ff6c-f9ea-4e93-bf0e-68f0cfab0c8a.png)

후위선택방식 과정을 보았을 때 가장 정확도가 떨어지지않고 특징값이 적은 부분을 선택했다.

![image](https://user-images.githubusercontent.com/26988563/164879263-c2a135d8-3ea7-4a44-9a57-3b8ae0839bf8.png)

후위제거방식으로 선택된 컬럼값을 x데이터에 적용시킨 결과 10개의 컬럼만 남은 것을 확인 가능하다.

![image](https://user-images.githubusercontent.com/26988563/164879275-ae33b1e5-5a0a-48fa-8ec0-152f71a190a6.png)

모델선택과 속성선택이 끝났으니 마지막으로 파라미터 튜닝단계를 실행하기 위해 데이터의 특성과 RF의 파라미터 설명들을 기반하여 어떤 파라미터 범위가 적절할지 범위를 크게 잡아 후보풀을 제작했다.

![image](https://user-images.githubusercontent.com/26988563/164879278-81e45561-7258-48fd-be59-cc4337481309.png)

피팅을 세네시간쯤 진행한 결과 진척상황이 보이지 않아 중단하기로 결정했다. 

![image](https://user-images.githubusercontent.com/26988563/164879282-d5114a4a-4317-46c7-b684-1152b41d7740.png)

파라미터 개수를 이유있게 줄여보고 교차검증 수도 10에서 5로 절반을 감소시킨 결과 58320에서 4320task로 줄었으며 결과도 훨씬 빨리 나오게 되었다. 아래 튜닝결과에 따라 찾고자 하는 범위를 줄여나가기로 했다.

![image](https://user-images.githubusercontent.com/26988563/164879287-8a8b7569-a8f5-4e4d-b59a-5ab276016d14.png)

두 번째 튜닝은 첫 번째 튜닝결과에서 숫자 범위를 좁혀나가고 부트스트랩이 참일 때 적용가능한 파라미터들을 실험해보았다.

![image](https://user-images.githubusercontent.com/26988563/164879293-d007c074-5411-4163-94d8-4306d5f442c7.png)

세 번째 파라미터 튜닝 결과이다. 갈수록 범위가 줄어드는 것을 확인 가능하다. 적정값이 나왔다고 해서 제일 최적의 결과에 가까운지를 확신할 수 없으니 업다운의 개념에 따라 적정값을 찾기위해 범위를 계속 조정해 나가며 튜닝을 진행한다.

![image](https://user-images.githubusercontent.com/26988563/164879298-3c9266c7-3f07-4cd3-9df1-523b2e3aad14.png)

모든 튜닝이 끝나고 테스트 셋을 불러와서 예측값을 도출해 낸다. 불러온 테스트셋을 전처리를 진행해주고 속성값을 버려주는 과정이다. 모델을 만든 x값 또한 전처리와 속성값이 선택되있으니 그에 맞춰준다.

![image](https://user-images.githubusercontent.com/26988563/164879307-49057b6f-196c-491e-b0a3-b2e39965688c.png)

튜닝된 모델에 준비된 테스트 셋을 적용시켜 예측값을 도출해 냈다.

![image](https://user-images.githubusercontent.com/26988563/164879311-7a2d7034-eb2e-4316-af1d-5fce54c8996a.png)

예측값을 csv파일 저장을 위해 데이터프레임으로 변환해주고 컬럼값을 지정해준다. 그 후에 csv로 저장한다.

![image](https://user-images.githubusercontent.com/26988563/164879316-10906854-7b05-4d27-86a2-6f2eb4d6ce6a.png)

제출양식에 맞춰 인덱스를 지워줬다. 더 나은 방법이 있을 것 같긴 한데 찾지 못하여 조금 원시적인 방식을 사용했다.

![image](https://user-images.githubusercontent.com/26988563/164879324-183254e9-678a-4f3e-93b4-f266c7d9dd13.png)

![image](https://user-images.githubusercontent.com/26988563/164879327-67486356-53aa-4a74-b4af-a818f5e02d6d.png)

제출시 데이터양 오류가 떴는데 지금까지 컬럼값이라 생각했던 것이 데이터라는 것을 마지막에 알게되었다. 여러 가지 방법을 시도해보다가 모두 실패하여 결국 데이터셋파일 자체를 고치는 방식을 사용하기로 했다.

![image](https://user-images.githubusercontent.com/26988563/164879341-1ae837b0-aab2-4999-86eb-783ae0224b57.png)

처음 파일을 올리고 테스트 셋은 트레이닝 셋과 다르다는 생각에 이것저것 적용해가며 올려봤지만 결국은 2순위에 그쳤다.

정확도가 93이상으로 올라가지 않는 이유가 궁금하다. 트레이닝 셋과 데이터셋의 차이 때문에 나오는 어쩔 수 없는 현상인지 아니면 더 개선될 여지가 있는데 학생레벨에서는 모르는 것인지가 궁금하다. 모델을 만드는 과정에서 수업에서 배운 것 이외에도 수많은 고급스킬들이 많음을 알게 되었고, 예측값을 모른다는 불확실성과 튜닝과정의 기다려야 하는 시간이 가장 어려웠던 대회였다. 하지만 튜닝을 하면서 초기의 정확도가 갈수록 조금씩이라도 증가하는 것을 보니 계속해서 욕심이 나고 더 좋은 방법을 부여하여 모델의 완성도를 높이고자 하는 것이 원동력이었다.

* Source code

```
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
bestmodel=RandomForestClassifier(n_estimators=500,max_depth=17,min_samples_split=2,min_samples_leaf=2,max_leaf_nodes=None,oob_score=True,random_state=1234)
bestmodel.fit(df_x,df_y)
testdf=pd.read_csv('C:/test_open.csv')
testdf_columns=df.loc[:,df.columns!='1.3']
scaled=scaler.transform(testdf)
testdf=pd.DataFrame(scaled, columns=testdf_columns.columns)
testdf=testdf[list(sfs2.k_feature_names_)]
pred_y=bestmodel.predict(testdf)
pred_y=pd.DataFrame(pred_y)
pred_y.columns=['answer']
pred_y.to_csv('D:/32144107_전세호.csv')
```


