# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#과제1구구단
lev=input()

for x in range(1,10):
    print('{0} X {1} = {2}'.format(lev, x, int(lev)*x))

#과제2BMI
tall=input('키 입력: ')
wgt=input('무게 입력: ')

BMI=float (wgt) / ((float(tall)/100)**2)

if(BMI<18.5):
    print('저체중')
elif(BMI<23):
    print('정상')    
elif(BMI<25):
    print('과체중')
elif(BMI<30):
    print('비만')
else:
    print('고도비만')    

#과제3 트리출력1    
height=input('높이를 입력하시오: ')

for x in range(1,int(height)+1):
    print(x*'*')

#과제4 트리출력2
height=input('높이를 입력하시오: ')

def starprint(height):
    for x in range(1,int(height)+1):
        print((int(height)-x)*' '+x*'*')

starprint(height)

#과제5 배열정렬
import numpy as np

before_arr=np.array([7,1,10,4,6,9,2,8,15,12,17,19,18])
after_arr=np.zeros(before_arr.size)

for x in range(1,before_arr.size+1):
    change_index=np.argmin(before_arr)
    after_arr[x-1]=before_arr[change_index].copy()
    before_arr[change_index]=999

print(after_arr)

#과제6 배열곱
import numpy as np

A=np.array([14,5,25])
B=np.array([7,35,87])

def mul_arr(A,B):
    C=np.zeros(A.size)
    C=A*B
    return C

print(mul_arr(A, B))

#과제7 배열생성
import numpy as np

my_arr=np.random.randint(1,50,50)
print(my_arr)

my_arr=np.reshape(my_arr,(10,5))
print('reshape 결과\n', my_arr)

#과제8 배열곱,슬라이싱
print('배열곱 결과\n',my_arr*2)

my_arr[my_arr<21]+=100
print('인덱싱 결과\n',my_arr)

print(my_arr[:,1:3])

print(my_arr[4:8,:])

#과제9 그래프 작성
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x=np.arange(12)
y=np.array([20,22,37,79,90,109,288,277,140,50,48,19])

plt.plot(x,y,color='red',label='rainfall')
plt.legend(loc='upper right')
plt.title('Month Raining')
plt.xlabel('month')
plt.ylabel('mm')
plt.grid(True)
plt.show()






































