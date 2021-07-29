# 慢特征图像变化检测
import numpy as np
import cv2
from scipy.linalg import inv, sqrtm, eig
import matplotlib.pyplot as pyplot

from sklearn.cluster import KMeans
import numpy.matlib
import matplotlib.pyplot as plt
from utility.metrics import Metrics_3class,Metrics_2class
from utility.envi_read import ENVI_read
import time
class SFA:
    def __init__(self, after, before):
        self.after = after
        self.before = before
        self.shape = None

    def process(self):
        after = self.after
        before = self.before

        self.shape = after.shape
        (rows, cols, bands) = self.shape

        # 归一化处理

        after = np.transpose(np.reshape(after, (rows * cols, bands)), (1, 0))
        before = np.transpose(np.reshape(before, (rows * cols, bands)), (1, 0))

        # 执行标准化程序
        after = self.standardization(after)
        before = self.standardization(before)
        e = np.cov(after - before)
        sum1 = np.sum((after[0] - before[0]) @ (after[0] - before[0]).T) / (rows * cols)
        sum2 = np.sum((after[1] - before[1]) @ (after[1] - before[1]).T) / (rows * cols)
        # print(after)
        e_x = np.cov(after)
        e_y = np.cov(before)
        B = 1 / 2 * (e_x + e_y)
        # 特征值与特征向量
        # print(B)
    
        (value, vector) = eig(np.linalg.inv(B) @ e)
        SFA = (vector @ after) - (vector @ before)
        # print(SFA)
        tr_sfa = np.transpose(SFA, (1, 0))
        
        # re = np.reshape(T, (j, 1))
        ##卡方距离

        #欧氏距离
        EuclideanMAD=np.sqrt(np.sum(tr_sfa[:,3:7]*tr_sfa[:,3:7],axis=1))#像素在各个波段统一变化强度
        # EuclideanMAD=np.sqrt(np.sum(tr_sfa*tr_sfa,axis=1))#像素在各个波段统一变化强度
        EuclideanMAD = np.reshape(EuclideanMAD, (rows*cols,1))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(EuclideanMAD)

        change_map = np.reshape(kmeans.labels_, (rows, cols,)).astype('uint8')

        return change_map

    def standardization(self, data):
        (rows, cols, bands) = self.shape
        data_mean = np.mean(data, axis=1)
        data_var = np.var(data, axis=1)
        data_mean_repeat = np.tile(data_mean, (rows * cols, 1)).transpose(1, 0)
        data_var_repeat = np.tile(data_var, (rows * cols, 1)).transpose(1, 0)
        data = (data - data_mean_repeat) / data_var_repeat
        # print(data)
        return data

import gdal
import scipy.io as sio
if __name__ == "__main__":
    tic=time.time()

    file_path_x='../data/Taizhou/2000TM'
    file_path_y='../data/Taizhou/2003TM'

    dataset_x = ENVI_read(file_path_x)
    print('shape of data_x:',dataset_x.XSize,dataset_x.YSize,dataset_x.im_bands)
    X=np.empty((dataset_x.XSize,dataset_x.YSize,dataset_x.im_bands),dtype=np.float32)#the data of envi

    for i in range(1,dataset_x.im_bands+1):
        band = i
        X[:,:,i-1] = dataset_x.get_data(band)  # 获取第n个通道的数据
        # plt.imshow(np.uint8(data))
        # plt.show()
        # print(band)
    dataset_y = ENVI_read(file_path_y)
    Y=np.empty((dataset_y.XSize,dataset_y.YSize,dataset_y.im_bands),dtype=np.float32)#the data of envi
    for i in range(1,dataset_y.im_bands+1):
        band = i
        Y[:,:,i-1] = dataset_y.get_data(band)  # 获取第n个通道的数据
    rows,cols,bands = X.shape
    ChangeRI=cv2.imread('../data/Taizhou/TaizhouChange_blackWhite.bmp',0)
    UnchangeRI=cv2.imread('../data/Taizhou/TaizhouChange_blackWhite_unchange.bmp',0)
    sfa = SFA(X, Y)
    change_map=sfa.process()
    ChangeMap=change_map
    CM=ChangeMap.copy()
    ChangeMap[ChangeMap==1]=255
    print('time:::::', time.time()-tic)
    cv2.imwrite('saf_bayarea.png',ChangeMap)
    #数值有三种：0,1,2
    AllRef =  np.zeros((rows, cols))
    for i in range(0,rows):
        for j in range(0,cols):
            if UnchangeRI[i,j]==255:
                AllRef[i,j]=1
            if ChangeRI[i,j]==255:
                AllRef[i,j]=2
    print('Calculating the evaluation criteria, please waiting...')
    metrics = Metrics_3class(ChangeMap//255,AllRef)
    metrics.evaluation()