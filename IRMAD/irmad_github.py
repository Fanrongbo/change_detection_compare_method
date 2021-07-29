import gdal
import numpy as np
from numpy.linalg import inv, eig
from scipy.stats import chi2
import cv2 as cv
from sklearn.cluster import KMeans
from utility.envi_read import ENVI_read
from utility.operation import Operation
from utility.metrics import Metrics_3class,Metrics_2class
import time
# from compare_method.sfa import get_binary_change_map
# from compare_method.covw import covw
import cv2
import scipy.io as sio
import matplotlib.pyplot  as plt
from utility import utils
def covw(x,w):
    if len(x.shape)>2:
        print('x must be 1- or 2-D.')
    if ((w>=0).all())==False:
        print('Weights must be nonnegative!')
    else:
        x=x    #input 1-D or 2-D
        w=w    #weight观测值用权值向量W进行加权，所有的权值都是非负值。COVW(X,W)给出方差-协方差矩阵的加权估计。
        # self.varargin#COVW(X,W,1)通过N进行归一化并产生关于其均值的观察值的二阶矩矩阵。COVW(X,W,0)等于COVW(X,W)
    m,n=x.shape
    mw,nw=w.shape
    
    # print((w>0).all())
    flag=0
    if m == 1:
        dispersion=0
    else:
        sumw=np.sum(w)
        # aa=np.tile(w,(1,n))
        meanw=np.sum(np.tile(w,(m,1))*x,axis=0)/sumw#根据x灰度值确定权重
        # meanw=np.sum(w*x)
        xc=x-np.tile(meanw,(m,1))# Remove weighted mean
        xc=np.tile(np.sqrt(w),(m,1))*xc
        dispersion=np.dot(xc,xc.T)/sumw
        if flag==0:
            dispersion = m/(m-1)*dispersion
    return dispersion,xc
def IRMAD(img_X, img_Y, max_iter=50, epsilon=1e-3):

    print('shape',img_X.shape)
    tic=time.time()
    bands_count_X, img_height, img_width = img_X.shape
    img_X = np.reshape(img_X, (-1, img_height * img_width))
    img_Y = np.reshape(img_Y, (-1, img_height * img_width))
    
    bands_count_X = img_X.shape[0]

    weight = np.ones((1, img_height * img_width))  # (1, height * width)
    can_corr = 100 * np.ones((bands_count_X, 1))
    for iter in range(max_iter):
        mean_X = np.sum(weight * img_X, axis=1, keepdims=True) / np.sum(weight)
        mean_Y = np.sum(weight * img_Y, axis=1, keepdims=True) / np.sum(weight)

        # centralization
        center_X = img_X - mean_X
        center_Y = img_Y - mean_Y
        # print(np.var(center_X,axis=1))#对角的数值很大，对角
        # print(np.cov(center_Y))#对角的数值很大 
        
        # also can use np.cov, but the result would be sightly different with author' result acquired by MATLAB code
        input=np.concatenate((center_X, center_Y),axis=0)
        cov_XY,xc = covw(input, weight)
        xc1=xc[0:6,:]
        
        # print('meanw',xc1.shape)
        size = cov_XY.shape[0]
        sigma_11 = cov_XY[0:bands_count_X, 0:bands_count_X]
        sigma_22 = cov_XY[bands_count_X:size, bands_count_X:size]
        sigma_12 = cov_XY[0:bands_count_X, bands_count_X:size]
        sigma_21 = sigma_12.T

        target_mat = np.dot(np.dot(np.dot(inv(sigma_11), sigma_12), inv(sigma_22)), sigma_21)
        eigenvalue, eigenvector_X = eig(target_mat)  # the eigenvalue and eigenvector of image X
        # sort eigenvector based on the size of eigenvalue
        eigenvalue = np.sqrt(eigenvalue)

        idx = (eigenvalue).argsort()#特征值从小到大,特征值越小越不相关，变化越大
        eigenvalue = eigenvalue[idx]
        
        if (iter + 1) == 1:
            print('Canonical correlations')
        # print(eigenvalue)
        eigenvector_X = eigenvector_X[:, idx]
        
        target_mat = np.dot(np.dot(np.dot(inv(sigma_22), sigma_21), inv(sigma_11)), sigma_12)
        eigenvaluey, eigenvector_Y = eig(target_mat)  # the eigenvalue and eigenvector of image X#A和B的特征值一样
        
        # sort eigenvector based on the size of eigenvalue
        eigenvaluey = np.sqrt(eigenvaluey)
        idx = eigenvaluey.argsort()
        eigenvector_Y = eigenvector_Y[:, idx]
        eigenvaluey=eigenvaluey[idx]

        # tune the size of X and Y, so the constraint condition can be satisfied调整X和Y的大小，以满足约束条件
        norm_X = np.sqrt(1 / np.diag(np.dot(eigenvector_X.T, np.dot(sigma_11, eigenvector_X))))
        norm_Y = np.sqrt(1 / np.diag(np.dot(eigenvector_Y.T, np.dot(sigma_22, eigenvector_Y))))
        
        eigenvector_X = norm_X * eigenvector_X
        eigenvector_Y = norm_Y * eigenvector_Y
        

        invstderr= np.diag(1./np.std(center_X,axis=1))#构成对角矩阵
        sgn=np.diag(np.sign(np.sum(invstderr@sigma_11@eigenvector_X,axis=1)))#确保x1和x1*a之间的正相关性之和为正
        eigenvector_X=eigenvector_X@sgn#matlab中*是@，.*是*，
        # # 确保规范变量对之间存在正相关
        eigenvector_Y=eigenvector_Y*np.diag(np.sign(eigenvector_X.T@sigma_12@eigenvector_Y))
        
        #MAD 变量的色散矩阵
        mad_variates = np.dot(eigenvector_X.T, center_X) - np.dot(eigenvector_Y.T, center_Y)  # (6, width * height)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('Iteration:',iter,' | eigenvalue',eigenvalue)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

        if np.max(np.abs(can_corr - eigenvalue)) < epsilon:
            print('########################break##########################')
            break
        
        # print('eigenvalue',iter,np.max(np.abs(can_corr - eigenvalue)),eigenvalue)
        can_corr = eigenvalue
        # calculate chi-square distance and probility of unchanged
        mad_var = np.reshape(2 * (1 - can_corr), (bands_count_X, 1))
        chi_square_dis = np.sum(mad_variates * mad_variates / mad_var, axis=0, keepdims=True)#计算波段之间的卡方距离        
        weight = 1 - chi2.cdf(chi_square_dis, bands_count_X)
        
    # for i in range(mad_variates.shape[0]):
        # cv2.imwrite('mad_%d.png'%i,mad_variates[i].reshape(img_height, img_width ).astype(np.uint8))
    if (iter + 1) == max_iter:
        print('the canonical correlation may not be converged')
    else:
        print('the canonical correlation is converged, the iteration is %d' % (iter + 1))
    toc=time.time()
    print('total time:',toc-tic)
    return mad_variates, can_corr, mad_var, eigenvector_X, eigenvector_Y, \
           sigma_11, sigma_22, sigma_12, chi_square_dis, weight

def get_binary_change_map(data, method='k_means'):
    """
    get binary change map
    :param data:
    :param method: cluster method
    :return: binary change map
    """
    if method == 'k_means':
        cluster_center = KMeans(n_clusters=2, max_iter=1500).fit(data.T).cluster_centers_.T  # shape: (1, 2)
        # cluster_center = k_means_cluster(weight, cluster_num=2)
        print('k-means cluster is done, the cluster center is ', cluster_center)
        dis_1 = np.linalg.norm(data - cluster_center[0, 0], axis=0, keepdims=True)
        dis_2 = np.linalg.norm(data - cluster_center[0, 1], axis=0, keepdims=True)

        bcm = np.copy(data)  # binary change map
        if cluster_center[0, 0] > cluster_center[0, 1]:
            bcm[dis_1 > dis_2] = 0
            bcm[dis_1 <= dis_2] = 255
        else:
            bcm[dis_1 > dis_2] = 255
            bcm[dis_1 <= dis_2] = 0
    elif method == 'otsu':
        bcm, threshold = otsu(data, num=400)
        print('otsu is done, the threshold is ', threshold)

    return bcm
if __name__ == '__main__':
    img_X = np.load("../data/river/river_1.npy")[:,:,:30]
    img_Y = np.load("../data/river/river_2.npy")[:,:,:30]
    img_width,img_height,channel=img_X.shape
    img_X=img_X.transpose((2,0,1))
    img_Y=img_Y.transpose((2,0,1))

    mad, can_coo, mad_var, ev_1, ev_2, sigma_11, sigma_22, sigma_12, chi2, noc_weight = IRMAD(img_X.astype(np.float32), img_Y.astype(np.float32), max_iter=26,epsilon=1e-4)
    sqrt_chi2 = np.sqrt(chi2)

    # EuclideanMAD=np.sqrt(np.sum(mad*mad,axis=0))#像素在各个波段统一变化强度

    EuclideanMAD = np.reshape(sqrt_chi2, (img_width*img_height,1))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(EuclideanMAD)
    ChangeMap = np.reshape(kmeans.labels_, (img_width, img_height,))
    # ChangeMap=abs(ChangeMap-1)
    ChangeMap = ChangeMap.astype(np.uint8)
    cv2.imwrite('ChangeMap.png',ChangeMap*255)
    #Change Label
    ChangeRI=cv2.imread('../data/river/river_ref.bmp',0)//255
    AllRef = ChangeRI
    img_shape=(img_width, img_height)
    print('Calculating the evaluation criteria, please waiting...')
    print(ChangeMap.shape,AllRef.shape)
    metrics = Metrics_2class(ChangeMap,AllRef)
    metrics.evaluation()





