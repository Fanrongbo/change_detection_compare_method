
import numpy as np


class Operation:
    # def __init__(self,x,w):
        
    def covw(self,x,w):
        if len(x.shape)>2:
            print('x must be 1- or 2-D.')
        if ((w>=0).all())==False:
            print('Weights must be nonnegative!')
        else:
            self.x=x    #input 1-D or 2-D
            self.w=w    #weight观测值用权值向量W进行加权，所有的权值都是非负值。COVW(X,W)给出方差-协方差矩阵的加权估计。
            # self.varargin#COVW(X,W,1)通过N进行归一化并产生关于其均值的观察值的二阶矩矩阵。COVW(X,W,0)等于COVW(X,W)
        m,n=x.shape
        mw,nw=w.shape
        
        # print((w>0).all())
        flag=0
        if m == 1:
            dispersion=0
        else:
            sumw=np.sum(w)
            aa=np.tile(w,(1,n))
            meanw=np.sum(np.tile(w,(1,n))*x,axis=0)/sumw#根据x灰度值确定权重
            xc=x-np.tile(meanw,(m,1))# Remove weighted mean
            xc=np.tile(np.sqrt(w),(1,n))*xc
            dispersion=np.dot(xc.T,xc)/sumw
            if flag==0:
                dispersion = m/(m-1)*dispersion
        return dispersion,meanw
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
# after = cv2.imread("Dubai_11122012.jpg")
# before = cv2.imread("Dubai_11272000.jpg")
# print('original size:',before.shape,after.shape)
###配准
# registration = SIFT(after,before)
# before_reg,_,_,_,_ = registration.siftImageAlignment(0.75)
# print('size after registration: ',before_reg.shape,after.shape)
# check_map=registration.checkboard(after,before_reg)
# pyplot.imshow(np.uint8(check_map))


# new_size = np.asarray(before.shape) / 5
# new_size = new_size.astype(int) * 5#将图像大小变为5的倍数
# row=min(after.shape[0],before_reg.shape[0])
# col=min(after.shape[1],before_reg.shape[1])
# print('2',row,col)
# before = before[0:row,0:col,:]
# after = after[0:row,0:col,:]

# before = cv2.resize(before, (new_size[1],new_size[0])).astype(np.int16)
# after = cv2.resize(after, (new_size[1],new_size[0])).astype(np.int16)
# print(after.shape,before.shape)