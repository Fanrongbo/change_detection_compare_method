import gdal
import numpy as np
from numpy.linalg import inv, eig
from scipy.stats import chi2
import cv2  
import matplotlib.pyplot as plt


img=np.load('river_1.npy')
print(img.shape)
img_show=img[:,:,1:4]
print(img_show.shape,np.max(img_show))
plt.figure(1)
plt.imshow(img_show/np.max(img_show))
# img_show=np.array(img_show*255,np.uint8)
# cv2.imwrite('img_show.png',img_show)

img=np.load('river_2.npy')
print(img.shape)
img_show=img[:,:,5:8]
plt.figure(2)
print(img_show.shape,np.max(img_show))
plt.imshow(img_show/np.max(img_show))
plt.show()