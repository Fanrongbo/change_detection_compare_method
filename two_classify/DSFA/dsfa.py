# -*- coding: utf-8 -*-
import argparse
import logging
import os
import cv2
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy import misc
from utility import utils
from utility.metrics import Metrics_3class,Metrics_2class
# import utils
from utility.model import dsfa

from libtiff import TIFF
from utility import preprocess
net_shape = [128, 128, 6] 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

def parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-e','--epoch',help='epoches',default=20000, type=int)
    parser.add_argument('-l','--lr',help='learning rate',default=5*1e-5, type=float)
    parser.add_argument('-r','--reg',help='regularization parameter',default=1e-4, type=float)
    parser.add_argument('-t','--trn',help='number of training samples',default=4000, type=int)
    parser.add_argument('-g','--gpu', help='GPU ID', default='0')
    parser.add_argument('--area',help='datasets', default='../data/river')
    args = parser.parse_args()

    return args

def main(img1, img2, args=None):
    #preprocess

    
    img_shape = np.shape(img1)#(463, 241, 198)
    print('img_shape',img_shape)
    im1 = np.reshape(img1, newshape=[-1,img_shape[-1]])
    im2 = np.reshape(img2, newshape=[-1,img_shape[-1]])

    im1 = utils.normlize(im1)
    im2 = utils.normlize(im2)
    
    
    # chg_ref = np.reshape(chg_map, newshape=[-1])

    imm = None
    all_magnitude = None
    # differ = np.zeros(shape=[np.shape(chg_ref)[0],net_shape[-1]])

    # load cva pre-detection result
    # ind = sio.loadmat(args.area+'/cva_ref.mat')
    cva_ind = cv2.imread('Taizhou_mad_result.png')[:,:,0]
    # cva_ind = ind['cva_ref']
    # plt.figure(2)
    # plt.imshow(cva_ind )
    # plt.show()
    # cv2.imwrite('./data/river/river_preprocess.png',np.abs(1-cva_ind)*255)

    cva_ind = np.reshape(cva_ind, newshape=[-1])
    
    i1, i2 = utils.getTrainSamples(cva_ind, im1, im2, args.trn)
    # print(i1.shape)
    # plt.show()
    loss_log, vpro, fcx, fcy, bval = dsfa(
        xtrain=i1, ytrain=i2, xtest=im1, ytest=im2, net_shape=net_shape, args=args)

    imm, magnitude, differ_map = utils.linear_sfa(fcx, fcy, vpro, shape=img_shape)

    magnitude = np.reshape(magnitude, img_shape[0:-1])
    # differ = differ_map

    change_map = np.reshape(utils.kmeans(np.reshape(magnitude, [-1])), img_shape[0:-1])
    
    # print('change_map',change_map.shape,'chg_map',chg_map.shape)
    # # magnitude
    # metrics = Metrics()
    # AllRef = np.load('../data/Taizhou/changemap.npy')
    # metrics.demo_eval(AllRef,1 - change_map,shape=img_shape,flag=1)
    # metrics.demo_eval(AllRef,  change_map, shape=img_shape, flag=2)
    # plt.imsave('results2.png',change_map, cmap='gray')
    # plt.figure(0)
    # plt.imshow(imm)
    # plt.figure(1)
    # plt.imshow(change_map)
    # # plt.title('change_map')
    # plt.figure(2)
    # plt.imshow(magnitude)
    # plt.show()

    return change_map


if __name__ == '__main__':
    args = parser()
    img1 = np.load('../data/Taizhou/2000TM.npy')
    img2 = np.load('../data/Taizhou/2003TM.npy')
    ChangeRI=cv2.imread('../data/Taizhou/TaizhouChange_blackWhite.bmp',0)
    UnchangeRI=cv2.imread('../data/Taizhou/TaizhouChange_blackWhite_unchange.bmp',0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    change_map=main(img1, img2,  args=args)
    (img_width, img_height,c)=img1.shape
    #数值有三种：0,1,2
    AllRef =  np.zeros((img_width, img_height))
    for i in range(0,img_width):
        for j in range(0,img_height):
            if UnchangeRI[i,j]==255:
                AllRef[i,j]=1
            if ChangeRI[i,j]==255:
                AllRef[i,j]=2
    cv2.imwrite('change_map.png',change_map*255)
    print('Calculating the evaluation criteria, please waiting...')
    metrics = Metrics_3class(change_map,AllRef)
    metrics.evaluation()