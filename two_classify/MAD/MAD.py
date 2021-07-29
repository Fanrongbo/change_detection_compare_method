import numpy as np
from scipy.linalg import inv, sqrtm, eig
import scipy
import matplotlib.pyplot as pyplot
import numpy.matlib
from sklearn.cluster import KMeans
import cv2
from utility.metrics import Metrics_3class,Metrics_2class
from utility.envi_read import ENVI_read
import matplotlib.pyplot as plt
import time
class MAD:
    def __init__(self,after,before):
        self.after = after
        self.before = before
    def process(self):
        after = self.after
        before=self.before
        (rows, cols, bands) = after.shape

        # 归一化处理
        after = np.transpose(np.reshape(after, (rows * cols, bands)), (1, 0))#对二维数组的transpose操作就是对原数组的转置操作。
        before = np.transpose(np.reshape(before, (rows * cols, bands)), (1, 0))
        
        con_cov = np.cov(after, before)#协方差矩阵单通道:(2,2),三通道:(6,6)
        
        cov_xx = con_cov[0:bands, 0:bands]
        cov_xy = con_cov[0:bands, bands:]
        cov_yx = con_cov[bands:, 0:bands]
        cov_yy = con_cov[bands:, bands:]
        # yy_cov = np.cov(before)
        # print('inv',inv(cov_xx),cov_xx)
        #注意：三通道数值相同会造成奇异矩阵
        A = inv(cov_xx) @ cov_xy @ inv(cov_yy) @ cov_yx #@是点乘！
        B = inv(cov_yy) @ cov_yx @ inv(cov_xx) @ cov_xy  # 与A特征值相同，但特征向量不同
        # print('a',A)
        # A的特征值与特征向量 av 特征值， ad 特征向量
        [av, ad] = eig(A)#A=λX
        # print('A:eigenvalue',av,ad,A)
        

        # 对特征值从小到大排列 与 CCA相反
        swap_av_index = np.argsort(av)
        # print('swap_av_index',av)
        # print('swap_av_index',ad)
        swap_av = av[swap_av_index[:av.size:1]]
        swap_ad = ad[swap_av_index[:av.size:1], :]
        # print('swap_av_index',av)
        # print('swap_av_index',swap_av_index,swap_av,swap_ad)
        # 满足st 条件
        ma = inv(sqrtm(swap_ad.T @ cov_xx @ swap_ad))  # 条件一

        swap_ad = swap_ad @ ma
        # print('after ad:',swap_ad)
        # 对应b的值
        [bv, bd] = eig(B)
        swap_bv = bv[swap_av_index[:bv.size:1]]
        swap_bd = bd[swap_av_index[:bd.size:1]]
        mb = inv(sqrtm(swap_bd.T @ cov_yy @ swap_bd))  # 条件二

        swap_bd = swap_bd @ mb
        # ab = np.linalg.inv(cov_yy) @ cov_yx @ swap_ad
        # bb = np.linalg.inv()

        MAD = swap_ad.T @ after - (swap_bd.T @ before)
        [i, j] = MAD.shape
        
        var_mad = np.zeros(i) 
        for k in range(i):
            var_mad[k] = np.var(MAD[k])
        var_mad = np.transpose(np.matlib.repmat(var_mad, j, 1), (1, 0))
        # print(var_mad)
        res = MAD * MAD / var_mad
        T = res.sum(axis=0) #通道轴相加
        # Kmeans 聚类
        re = np.reshape(T, (j, 1))
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(re)
        img = np.reshape(kmeans.labels_, (rows, cols,))
        img = img.astype(np.uint8)
        center = kmeans.cluster_centers_
        cv2.imwrite('ElephantButte_MAD.jpg', img*255)
        return np.abs(1-img)
class SIFT:
    def __init__(self,img1,img2):
        self.img1=img1
        self.img2=img2
    def sift_kp(self,image):
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d_SIFT.create()
        kp,des = sift.detectAndCompute(image,None)
        # kp_image = cv2.drawKeypoints(gray_image,kp,None)
        return kp,des
    def get_good_match(self,des1,des2,dist):
        bf = cv2.BFMatcher()
        # print(len(des1),len(des2))
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < dist * n.distance:
                good.append(m)
        return good
    def siftImageAlignment(self,dist):
        img1 = self.img1
        img2 = self.img2
        # print(img1.shape,img2.shape)
        kp1,des1 = self.sift_kp(img1)
        kp2,des2 = self.sift_kp(img2)
        # print(len(des1),len(des2))
        goodMatch = self.get_good_match(des1,des2,dist)
        print('goodMatch',len(goodMatch))
        
        ransacReprojThreshold = 4
        if len(goodMatch) > 4:
            ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
            imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # Y1=Y1/Y1[2]
        matchesMask = status.ravel().tolist()
        kp_num = matchesMask.count(1)
        kp1_matrix = np.zeros((3,kp_num), dtype="f")
        kp2_matrix = np.zeros((3,kp_num), dtype="f")
        i=0
        for m in range(len(matchesMask)):
            if matchesMask[m]==1:
                kp1_matrix[0][i] = ptsA[m][0][0]
                kp1_matrix[1][i] = ptsA[m][0][1]
                kp1_matrix[2][i] = 1
                kp2_matrix[0][i] = ptsB[m][0][0]
                kp2_matrix[1][i] = ptsB[m][0][1]
                kp2_matrix[2][i] = 1
                i = i+1
        kp1_transform = np.zeros((3,kp_num), dtype="f")
        for i in range(kp_num):
            X1=np.array([[kp1_matrix[0][i]],[kp1_matrix[1][i]],[1]])
            Y1=np.dot(H,X1)
            Y1=Y1/Y1[2]
            kp1_transform[0][i] = Y1[0][0]
            kp1_transform[1][i] = Y1[1][0]
            kp1_transform[2][i] = 1
        point_A = np.zeros(shape=(kp_num,2))
        point_B = np.zeros(shape=(kp_num,2))#变换点GPs
        for i in range(kp_num):
            point_A[i][0] = kp2_matrix[0][i]
            point_A[i][1] = kp2_matrix[1][i]
            point_B[i][0] = kp1_transform[0][i]
            point_B[i][1] = kp1_transform[1][i]
        error = kp2_matrix - kp1_transform
        error = error[0:2].transpose()
            #cumpute mean square
        mean_square = np.sum(np.square(error),axis=1)
        rmse = np.sqrt(np.sum(mean_square)/float(kp_num))
        print('emse',rmse)
        return imgOut,H,status,point_A,point_B
    def checkboard(self,I1, I2, n=7):
        assert I1.shape == I2.shape
        height, width, channels = I1.shape
        hi, wi = int(height/n), int(width/n)
        outshape = (int(hi*n), int(wi*n), int(channels))
        out_image = np.zeros(outshape, dtype='uint8')
        for i in range(n):
            h = hi * i
            h1 = h + hi
            for j in range(n):
                w = wi * j
                w1 = w + wi
                if (i-j)%2 == 0:
                    out_image[h:h1, w:w1, :] = I1[h:h1, w:w1, :]
                else:
                    out_image[h:h1, w:w1, :] = I2[h:h1, w:w1, :]
        return out_image
import scipy.io as sio

if __name__ == "__main__":
    tic = time.time()
    file_path_x='../data/Taizhou/2000TM'
    file_path_y='../data/Taizhou/2003TM'
    dataset_x = ENVI_read(file_path_x)
    print('shape of data_x:',dataset_x.XSize,dataset_x.YSize,dataset_x.im_bands)
    X=np.empty((dataset_x.XSize,dataset_x.YSize,dataset_x.im_bands),dtype=np.float32)#the data of envi
    for i in range(1,dataset_x.im_bands+1):
        band = i
        X[:,:,i-1] = dataset_x.get_data(band)  # 获取第n个通道的数据
    dataset_y = ENVI_read(file_path_y)
    Y=np.empty((dataset_y.XSize,dataset_y.YSize,dataset_y.im_bands),dtype=np.float32)#the data of envi
    for i in range(1,dataset_y.im_bands+1):
        band = i
        Y[:,:,i-1] = dataset_y.get_data(band)  # 获取第n个通道的数据
    ChangeRI=cv2.imread('../data/Taizhou/TaizhouChange_blackWhite.bmp',0)
    UnchangeRI=cv2.imread('../data/Taizhou/TaizhouChange_blackWhite_unchange.bmp',0)
    rows,cols,bands = X.shape
    irmad = MAD (X,Y)
    change_map=irmad.process()
    ChangeMap=change_map
    CM=ChangeMap.copy()
    ChangeMap[ChangeMap==1]=255
    print('time:::::', time.time()-tic)
    cv2.imwrite('mad_taizhou.png',ChangeMap)

    
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