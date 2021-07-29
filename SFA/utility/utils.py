# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
from matplotlib import image
from scipy.cluster.vq import kmeans as km
from sklearn.metrics import auc,roc_curve,accuracy_score,cohen_kappa_score,classification_report,confusion_matrix
from matplotlib import pyplot as plt


def metric(img=None, chg_ref=None,flag=0):
    shape = img.shape
    chg_ref = np.array(chg_ref, dtype=np.float32)
    chg_ref = chg_ref / np.max(chg_ref)

    plt.figure(66)
    plt.imshow(chg_ref.reshape(shape[0], shape[1]))
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)  # 设置色标刻度字体大小。
    plt.savefig('label_%d.png' % flag)
    plt.clf()


    confusion_map = np.zeros((shape[0] * shape[1]))

    img = np.reshape(img, [-1])
    chg_ref = np.reshape(chg_ref, [-1])

    N=shape[0] * shape[1]
    for i in range(0, N):
        if img[i] == 1 and chg_ref[i] == 1:
            confusion_map[i] = 1  # TP
        if img[i] == 0 and chg_ref[i] == 0:
            confusion_map[i] = 2  # TN
        if img[i] == 0 and chg_ref[i] == 1:
            confusion_map[i] = 3  # FN
        if img[i] == 1 and chg_ref[i] == 0:
            confusion_map[i] = 4  # FP

    plt.figure(56)
    plt.imshow(confusion_map.reshape(shape[0], shape[1]), cmap='jet')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)  # 设置色标刻度字体大小。
    plt.savefig('confusion_map_%d.png' % flag)
    plt.clf()


    loc1 = np.where(chg_ref == 1)[0]
    num1 = np.sum(img[loc1] == 1)#change right
    acc_chg = np.divide(float(num1), float(np.shape(loc1)[0]))

    loc2 = np.where(chg_ref == 0)[0]
    num2 = np.sum(img[loc2] == 0)#unchange right
    acc_un = np.divide(float(num2), float(np.shape(loc2)[0]))

    acc_all = np.divide(float(num1 + num2), float(np.shape(loc1)[0] + np.shape(loc2)[0]))

    loc3 = np.where(img == 1)[0]
    num3 = np.sum(chg_ref[loc3] == 1)
    acc_tp = np.divide(float(num3), float(np.shape(loc3)[0]))
    
    print('采样样本总数(变化总数+未变化总数)：',(np.shape(loc1)[0] + np.shape(loc2)[0]),loc1.shape[0],loc2.shape[0])
    # print('')
    print('Accuracy of Unchanged Regions is: %.4f' % (acc_un))
    print('Accuracy of Changed Regions is:   %.4f' % (acc_chg))
    print('The True Positive ratio is:       %.4f' % (acc_tp))
    print('Accuracy of all testing sets is : %.4f' % (acc_all))
    lastRef=chg_ref.copy()
    lastRef=lastRef.ravel()
    test=img.copy()
    test=test.ravel()
    # print('change_map shape',lastRef.shape)
    # print('采样样本总数(变化总数+未变化总数)：',lastRef.shape[0],np.sum(test==1),np.sum(test==0))
    tn, fp, fn, tp = confusion_matrix(lastRef, test).ravel()
    fpr, tpr, thresholds = roc_curve(lastRef, test )
    auc_value=auc(fpr, tpr)
    kap=cohen_kappa_score(lastRef, test)
    accc=accuracy_score(lastRef, test)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('[tn, fp, fn, tp]:',tn, fp, fn, tp)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('OA:',accc,'| kappa:',kap,' | AUC:',auc_value)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    target_names = ['unchange', 'change']
    print(classification_report(lastRef, test, target_names=target_names))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    plt.figure(55)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('Positive_Rate_%d.png' % flag)
    plt.clf()
    return acc_un, acc_chg, acc_all, acc_tp


def getTrainSamples(index, im1, im2, number=4000):

    loc = np.where(index != 1)[0]
    # print('loc',loc,len(loc))
    perm = np.random.permutation(np.shape(loc)[0])
    # print('perm',perm,len(perm))
    ind = loc[perm[0:number]]
    # print('ind',ind)
    return im1[ind, :], im2[ind, :]


def normlize(data):
    meanv = np.mean(data, axis=0)
    stdv = np.std(data, axis=0)

    delta = data - meanv
    data = delta / stdv

    return data


def chi_square(fcx, fcy, vp, shape):

    delta = np.matmul(fcx, vp) - np.matmul(fcy, vp)

    delta = delta**2 / np.std(delta, axis=0)

    delta = np.sqrt(delta)

    differ_map = delta#normlize(delta)
    
    magnitude = np.sum(delta, axis=1)
    print('shape',differ_map.shape,magnitude.shape)
    vv = magnitude / np.max(magnitude)

    im = np.reshape(kmeans(vv), shape[0:-1])

    return im, magnitude, differ_map

def linear_sfa(fcx, fcy, vp, shape):

    delta = np.matmul(fcx, vp) - np.matmul(fcy, vp)
    # # print('shape',fcx.shape,vp.shape)#(111583, 6) (6, 6)
    #delta = delta / np.std(delta, axis=0)

    delta = delta**2

    differ_map = delta#normlize(delta)#(111583, 6)

    magnitude = np.sum(delta, axis=1)#(111583,)
    # # print('shape',differ_map.shape,magnitude.shape) 

    vv = magnitude / np.max(magnitude)

    im = np.reshape(kmeans(vv), shape[0:-1])

    return im, magnitude, differ_map


def data_loader(area=None):

    img1_path = area + '/img_t1.mat'
    img2_path = area + '/img_t2.mat'
    change_path = area + '/chg_ref.bmp'

    mat1 = sio.loadmat(img1_path)
    mat2 = sio.loadmat(img2_path)

    img1 = mat1['im']
    img2 = mat2['im']

    chg_map = image.imread(change_path)

    return img1, img2, chg_map


def kmeans(data):
    shape = np.shape(data)
    # print((data))
    ctr, _ = km(data, 2)

    for k1 in range(shape[0]):
        if abs(ctr[0] - data[k1]) >= abs(ctr[1] - data[k1]):
            data[k1] = 0
        else:
            data[k1] = 1
    return data
