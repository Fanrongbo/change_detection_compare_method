
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