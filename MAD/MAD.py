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
from utility import utils

class MAD:
    def __init__(self,after,before):
        self.after = after
        self.before = before
    def process(self):
        #进行相应的处理计算
        # dataset_after = gdal.Open(
        #     r"F:\deeplearndata\rssrai2019_change_detection\train\train\img_2017\image_2017_960_960_8.tif")
        # im_width = dataset_after.RasterXSize  # 栅格矩阵的列数
        # im_height = dataset_after.RasterYSize  # 栅格矩阵的行数
        # im_bands = dataset_after.RasterCount  # 波段数
        # after = np.transpose(dataset_after.ReadAsArray(0, 0, im_width, im_height), (1, 2, 0))  # 获取数据
        #
        # dataset_before = gdal.Open(
        #     r"F:\deeplearndata\rssrai2019_change_detection\train\train\img_2018\image_2018_960_960_8.tif")
        # im_width = dataset_before.RasterXSize  # 栅格矩阵的列数
        # im_height = dataset_before.RasterYSize  # 栅格矩阵的行数
        # im_bands = dataset_before.RasterCount  # 波段数
        # before = np.transpose(dataset_before.ReadAsArray(0, 0, im_width, im_height), (1, 2, 0))  # 获取数据
        # after = np.expand_dims(self.after[:,:,0],axis=2)
        # before = np.expand_dims(self.before[:,:,0],axis=2)
        after = self.after
        before=self.before
        print(after.shape,before.shape)
        (rows, cols, bands) = after.shape

        # 归一化处理

        after = np.transpose(np.reshape(after, (rows * cols, bands)), (1, 0))#对二维数组的transpose操作就是对原数组的转置操作。
        before = np.transpose(np.reshape(before, (rows * cols, bands)), (1, 0))
        # after_mean = np.mean(after, axis=1)
        # after_var = np.std(after, axis=1)#标准差
        # before_mean = np.mean(before, axis=1)
        # before_var = np.std(before, axis=1)

        # print('after.shape',after.shape)
        
        # cov_aa_mari = np.cov(after)#协方差
        # print(cov_aa_mari)
        # print(type(cov_aa_mari),cov_aa_mari.shape,np.linalg.det(cov_aa_mari) )
        # cov_aa_mat_i = np.linalg.inv(cov_aa_mari)
        # print(after.shape,before.shape)
        con_cov = np.cov(after, before)#协方差矩阵单通道:(2,2),三通道:(6,6)
        
        # print('con_cov',con_cov.shape)
        cov_xx = con_cov[0:bands, 0:bands]
        cov_xy = con_cov[0:bands, bands:]
        cov_yx = con_cov[bands:, 0:bands]
        cov_yy = con_cov[bands:, bands:]
        # print('cov_xx',cov_xx,'aa',np.cov(after),'aa',np.var(after[0,:]),'aa',np.var(after[1,:]))
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
        print('MAD',MAD.shape)
        [i, j] = MAD.shape
        
        var_mad = np.zeros(i) 
        for k in range(i):
            var_mad[k] = np.var(MAD[k])
        var_mad = np.transpose(np.matlib.repmat(var_mad, j, 1), (1, 0))
        # print(var_mad)
        res = MAD * MAD / var_mad
        T = res.sum(axis=0) #通道轴相加
        print('T.shape',T.shape)
        # Kmeans 聚类
        re = np.reshape(T, (j, 1))
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(re)
        img = np.reshape(kmeans.labels_, (rows, cols,))
        # print(img.shape,type(img))
        # img=np.abs(img-1)
        # img = np.expand_dims(img,axis=2)
        img = img.astype(np.uint8)
        # print(np.max(img))
        # img = cv2.erode(img,kernel).astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素
        # img = cv2.imread('j_noise_out.bmp', 0)
        # print('aaaaaaaa',img.shape)
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
        # img = ndimage.binary_erosion(img)#腐蚀运算
        center = kmeans.cluster_centers_
        # print(np.max(img)) 
        # scipy.misc.imsave('MAD2.jpg', img)
        cv2.imwrite('ElephantButte_MAD.jpg', img*255)
        print('center of k_mean:',center)
        # pyplot.imshow(np.uint8(img))
        # pyplot.show()
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
import time
if __name__ == "__main__":
    tic = time.time()
    img_X = np.load("../data/river/river_1.npy")[:,:,:30]
    img_Y = np.load("../data/river/river_2.npy")[:,:,:30]
    ChangeRI=cv2.imread('../data/river/river_ref.bmp',0)//255

    img_width,img_height,channel=img_X.shape
    irmad = MAD (img_X,img_Y)
    change_map=irmad.process()
    ChangeMap=change_map
    CM=ChangeMap.copy()
    ChangeMap[ChangeMap==1]=255
    cv2.imwrite('Taizhou_mad_result.png',ChangeMap)
    print('time:::::', time.time()-tic)
    #Change Label
    AllRef = ChangeRI
    img_shape=(img_width, img_height)
    print('Calculating the evaluation criteria, please waiting...')
    print(ChangeMap.shape,AllRef.shape)
    metrics = Metrics_2class(ChangeMap//255,AllRef)
    metrics.evaluation()