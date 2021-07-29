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
import time
# import utils
from utility.model import dsfa
from utility.metrics import Metrics_3class,Metrics_2class
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

    tic=time.time()
    img_shape = np.shape(img1)#(463, 241, 198)
    print('img_shape',img_shape)
    im1 = np.reshape(img1, newshape=[-1,img_shape[-1]])
    im2 = np.reshape(img2, newshape=[-1,img_shape[-1]])

    im1 = utils.normlize(im1)
    im2 = utils.normlize(im2)

    imm = None
    all_magnitude = None

    # load cva pre-detection result
    ind = sio.loadmat(args.area+'/cva_ref.mat')
    cva_ind = ind['cva_ref']

    cva_ind = np.reshape(cva_ind, newshape=[-1])
    i1, i2 = utils.getTrainSamples(cva_ind, im1, im2, args.trn)

    loss_log, vpro, fcx, fcy, bval = dsfa(
        xtrain=i1, ytrain=i2, xtest=im1, ytest=im2, net_shape=net_shape, args=args)

    imm, magnitude, differ_map = utils.linear_sfa(fcx, fcy, vpro, shape=img_shape)
    magnitude = np.reshape(magnitude, img_shape[0:-1])
    # differ = differ_map

    change_map = np.reshape(utils.kmeans(np.reshape(magnitude, [-1])), img_shape[0:-1])
    # print('change_map',change_map.shape,'chg_map',chg_map.shape)
    # # magnitude
    # acc_un, acc_chg, acc_all2, acc_tp = utils.metric(1-change_map, chg_map,0)
    # acc_un, acc_chg, acc_all3, acc_tp = utils.metric(change_map, chg_map,1)
    # plt.imsave('results2.png',change_map, cmap='gray')
    # print('time:::::', time.time()-tic)

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
    ChangeRI=cv2.imread('../data/river/river_ref.bmp',0)//255
    img1=np.load('../data/river/river_1.npy')
    img2=np.load('../data/river/river_2.npy')
    print(img1.shape,type(img1),img1.dtype)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    change_map=main(img1, img2, args=args)
    (img_width, img_height,c)=img1.shape
    
    AllRef = ChangeRI
    img_shape=(img_width, img_height)
    print('Calculating the evaluation criteria, please waiting...')
    metrics = Metrics_2class(change_map,AllRef)
    metrics.evaluation()