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



















































































