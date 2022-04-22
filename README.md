# Univ_ML
(ML_modelmaking은 별개의 작업입니다.)

여러 날씨이미지 데이터셋에서 어떤 날씨인지를 구별해내는 딥러닝 모델을 만드는 것이 목표입니다.

![image](https://user-images.githubusercontent.com/26988563/164728375-fac6c30a-f4dd-4ebb-937b-842f6a1bf388.png)

각 이미지별로 분류파일을 생성해 분류한 뒤 확장자를 jpg로 통일했다.

![image](https://user-images.githubusercontent.com/26988563/164728421-c2582c5b-9276-4d3f-a113-126d22c18ab8.png)

![image](https://user-images.githubusercontent.com/26988563/164728429-35eb6604-0f33-4325-bed9-9216f3aee338.png)

학습데이터가 적은 편이기에 datagen을 사용해 하나의 사진당 20개의 추가 이미지파일을 생성했다.

![image](https://user-images.githubusercontent.com/26988563/164728498-6256f3dc-4af2-48b4-a0b4-16f1229be902.png)

학습할 이미지가 날씨이기 때문에 이미지의 크기가 작은 편이 학습과정을 가볍게 만들어주고 특징값을 더 쉽게 뽑아내며 효율적인 학습을 할 것이라 판단했다. 20개의 추가이미지데이터 생성을 위한 datagen인자들은 날씨이미지의 특징을 생각하여 구성했다.

![image](https://user-images.githubusercontent.com/26988563/164728547-05ff6a31-ce08-4871-b13d-6373a5e28c07.png)

모델은 최대한 간단하게 만들었는데 날씨이미지는 큰 특징이 아니라 적은 특징들로 분류를 해야 한다는 판단 하에 복잡도를 최대한 줄여서 날씨이미지분류에 정확하게 필요한 몇 가지의 특징들에만 집중해 분류를 시행하는 방향으로 모델을 작성했다.

![image](https://user-images.githubusercontent.com/26988563/164728610-f0a1cd2d-fc74-4956-824f-48c50028f733.png)

그렇게 해서 작성된 모델로 과적합을 테스트해보기 위해 에폭을 100으로 두고 테스트해본 결과 45언저리에서 과적합이 서서히 시작되는 것을 예측해볼 수 있었다.

![image](https://user-images.githubusercontent.com/26988563/164728664-ac7f999d-f2d2-436a-9945-e2afbcbaae2e.png)

예상대로 훈련정확도는 97~98에 육박하는 반면 예측정확도는 92에 그쳤다.

![image](https://user-images.githubusercontent.com/26988563/164728706-868e14cd-e1ce-4d76-b4e9-f6cfa594c12f.png)

그래서 정확히 45에서 끊어봤더니 또 정확도가 비슷하게 나오는 결과를 보고 다시 에폭을 55까지 늘려서 시행하여 완성된 모델을 얻을 수 있었다.

* neural network structure

![image](https://user-images.githubusercontent.com/26988563/164728894-e8a19897-36c4-4253-aae4-9ffb465e7844.png)

* Source code

```
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:34:39 2021
@author: tmark
"""
from PIL import Image
import sys,os,glob,numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator,img_to_array
sys.path.append(os.path.abspath('D:\\'))
import trainplot
#데이터 경로와 카테고리,원핫코딩을 위한 변수
data_dir='D:\dataset'
category=['cloudy','rain','shine','sunrise']
catesize=len(category)
#날씨는 크기가 작은게 유리하다고 판단
image_w,image_h=64,64
x=[]
y=[]
datagen=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
#이미지를 데이터셋화+카테고리별 onehot coding
for tag,cate in enumerate(category):
    #각 카테고리 별로 배열생성후 인덱스값에 1할당으로 원핫배열생성
    onehot=[0]*catesize
    onehot[tag]=1
    
    #각 카테고리별 이미지파일들을 glob을 사용해 읽어들임
    cate_dir=data_dir+'/'+cate
    files=glob.glob(cate_dir+'/*.jpg')
    
    #적은학습데이터를 위한 실시간 이미지증가
    for i,image in enumerate(files):
        img=Image.open(image)
        img=img.convert('RGB')
        img_plus=img_to_array(img)
        img_plus=img_plus.reshape((1,)+img_plus.shape)
        i=0
        for batch in datagen.flow(img_plus,batch_size=1,save_to_dir='D:\dataset\\'+cate,save_prefix='plus_'+cate,save_format='jpg'):
            i+=1
            if i>20:
                break
            
    files=glob.glob(cate_dir+'/*.jpg')
    #읽어들인 이미지파일들을 배열화하여 데이터로 저장
    for i,image in enumerate(files):
        img=Image.open(image)
        img=img.convert('RGB')
        img=img.resize((image_w,image_h))
        data=np.asarray(img)
        
        #카테고리별로 이미지데이터와 분류데이터 저장
        x.append(data)
        y.append(onehot)
        
#데이터를 분류, 나중을 대비해 저장
x=np.array(x,dtype=(float))
y=np.array(y,dtype=(float))
x=x.astype(float)/255
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.3,random_state=123)
checkdata=(x_tr,x_te,y_tr,y_te)
np.save('D:\dataset\checkdata.npy',checkdata)
#프로젝트 이어서 진행. 데이터 불러오기
x_tr,x_te,y_tr,y_te=np.load('D:\dataset\checkdata.npy',allow_pickle=True)
#모델 생성
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=x_tr.shape[1:],activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(catesize,activation='softmax'))
model.summary()
#모델학습및 학습과정 그래프출력
batch_size=20
epochs=55
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
disp=model.fit(x_tr,y_tr,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_te,y_te),callbacks=[trainplot.TrainingPlot()])
#테스트 데이터에 대한 예측값 출력
pred=model.predict(x_te)
classified=[np.argmax(y,axis=None,out=None)for y in pred]
score=model.evaluate(x_te,y_te,verbose=0)
print('Test loss:',score[0],'Test accuracy:',score[1])
#모델 저장및 불러오기
model.save('D:\dataset')
model.keras.models.load_model('D:\dataset')
```

* Test loss, Test accuracy

![image](https://user-images.githubusercontent.com/26988563/164729045-fba3f383-549c-4285-8332-237ceb9b5119.png)

* Training 마지막 5개 epoch 및 test loss/accuracy 출력 화면

![image](https://user-images.githubusercontent.com/26988563/164729191-e5ac2b3c-62f5-445a-a0c7-8a1f20c21cb2.png)

* 학습 곡선 그래프

![image](https://user-images.githubusercontent.com/26988563/164729252-a9528eb7-9523-43a4-9e1e-a7334889da94.png)


