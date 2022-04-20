# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 19:12:35 2021

@author: tmark
"""

import pandas as pd

cars=pd.read_csv('C:\/cars.csv')

#데이터셋 위쪽 5행
cars.head()
#데이터셋 컬럼 이름들
cars.columns
#데이터셋 두번째 컬럼 값
cars[cars.columns[1:]]
#데이터셋 11~20행 중 speed컬럼의 값
cars['speed'].iloc[11:21]
#speed 20~자료
fast=cars['speed']>19
cars[fast]
#speed 10~ and dist 50~인 행들의 자료
fast=cars['speed']>10
far=cars['dist']>50
cars[fast&far]
#speed 15~ and dist 50~인 행의 갯수
fast=cars['speed']>15
far=cars['dist']>50
cars[fast&far].count()











