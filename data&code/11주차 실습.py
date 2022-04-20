# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:45:56 2021

@author: tmark
"""
from skimage import io,color,transform
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import matplotlib.image as mp
import numpy as np
%matplotlib inline

fname=askopenfilename()
image=mp.imread(fname)
plt.imshow(image)

gray=color.rgb2gray(image)
plt.imshow(gray)

small_size=(image.shape[0]//3,image.shape[1]//3,image.shape[2])
small=transform.resize(image=image,output_shape=small_size)
plt.imshow(small)

plt.imshow(np.fliplr(image))
plt.imshow(np.flipud(image))






























