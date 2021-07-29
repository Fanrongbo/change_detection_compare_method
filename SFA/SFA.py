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
from utility import utils

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
        
        print('tr_sfa',tr_sfa.shape)
        # re = np.reshape(T, (j, 1))
        ##卡方距离
        # [i, j] = tr_sfa.shape
        # var_mad = np.empty(i) 
        # for k in range(i):
            # var_mad[k] = np.var(tr_sfa[k])
        # var_mad = np.transpose(np.matlib.repmat(var_mad, j, 1), (1, 0))
        # chi2_dis = tr_sfa * tr_sfa / var_mad
        # chi2_dis = chi2_dis.sum(axis=1) #通道轴相加
        # print('chi2_dis.shape',chi2_dis.shape)
        # chi2_dis = np.reshape(chi2_dis, (rows*cols,1))
        #欧氏距离
        EuclideanMAD=np.sqrt(np.sum(tr_sfa[:,3:7]*tr_sfa[:,3:7],axis=1))#像素在各个波段统一变化强度
        # EuclideanMAD=np.sqrt(np.sum(tr_sfa*tr_sfa,axis=1))#像素在各个波段统一变化强度
        EuclideanMAD = np.reshape(EuclideanMAD, (rows*cols,1))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(EuclideanMAD)

        change_map = np.reshape(kmeans.labels_, (rows, cols,)).astype('uint8')
        # change_map=abs(change_map-1).astype('uint8')
        
        # center = kmeans.cluster_centers_
        #tr_sfa = np.reshape(tr_sfa, (rows,cols,3))
        # pyplot.imshow(img)
        # pyplot.show()
        # print('change_map',change_map.shape)
        # kernel = np.ones((2,2), np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))#开运算
        # change_map = cv2.morphologyEx(change_map, cv2.MORPH_OPEN, kernel)
        # change_map = cv2.medianBlur(change_map, 3)#去除椒盐噪声，中值滤波
        # kernel     = np.asarray(((0,0,1,0,0),
                             # (0,1,1,1,0),
                             # (1,1,1,1,1),
                             # (0,1,1,1,0),
                             # (0,0,1,0,0)), dtype=np.uint8)
        # change_map = cv2.erode(change_map,kernel)#腐蚀
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
import time
if __name__ == "__main__":
    tic=time.time()
    img_X = np.load("../data/river/river_1.npy")[:,:,:30]
    img_Y = np.load("../data/river/river_2.npy")[:,:,:30]
    ChangeRI=cv2.imread('../data/river/river_ref.bmp',0)//255
    img_width,img_height,channel=img_X.shape
    sfa = SFA(img_X, img_Y)
    change_map=sfa.process()
    ChangeMap=change_map
    CM=ChangeMap.copy()
    ChangeMap[ChangeMap==1]=255
    cv2.imwrite('Taizhou_sfa_result.png',ChangeMap)
    # plt.title('Predict result') # 图像题目
    # cb = plt.colorbar()
    # plt.imshow(ChangeMap)
    
    print('time:::::', time.time()-tic)
    #Change Label
    AllRef = ChangeRI
    img_shape=(img_width, img_height)
    print('Calculating the evaluation criteria, please waiting...')
    print(ChangeMap.shape,AllRef.shape)
    metrics = Metrics_2class(ChangeMap//255,AllRef)
    metrics.evaluation()