# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)-15s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

def dsfa(xtrain, ytrain, xtest, ytest, net_shape=None, args=None):

    train_num = np.shape(xtrain)[0]
    bands = np.shape(xtrain)[-1]
    print('bands',xtrain.shape)
    tf.reset_default_graph()

    activation = tf.nn.softsign

    xd = tf.placeholder(dtype=tf.float32, shape=[None, bands])
    yd = tf.placeholder(dtype=tf.float32, shape=[None, bands])

    # fc1
    fc1w1 = tf.Variable(tf.truncated_normal(shape=[bands, net_shape[0]], dtype=tf.float32, stddev=1e-1))
    fc1w2 = tf.Variable(tf.truncated_normal(shape=[bands, net_shape[0]], dtype=tf.float32, stddev=1e-1))
    fc1b1 = tf.Variable(tf.constant(1e-1, shape=[net_shape[0]], dtype=tf.float32))
    fc1b2 = tf.Variable(tf.constant(1e-1, shape=[net_shape[0]], dtype=tf.float32))

    fc1x = tf.nn.bias_add(tf.matmul(xd, fc1w1), fc1b1)
    fc1y = tf.nn.bias_add(tf.matmul(yd, fc1w2), fc1b2)

    fc11 = activation(fc1x)
    fc12 = activation(fc1y)

    # fc2
    fc2w1 = tf.Variable(tf.truncated_normal(shape=[net_shape[0], net_shape[1]], dtype=tf.float32, stddev=1e-1))
    fc2w2 = tf.Variable(tf.truncated_normal(shape=[net_shape[0], net_shape[1]], dtype=tf.float32, stddev=1e-1))
    fc2b1 = tf.Variable(tf.constant(1e-1, shape=[net_shape[1]], dtype=tf.float32))
    fc2b2 = tf.Variable(tf.constant(1e-1, shape=[net_shape[1]], dtype=tf.float32))

    fc2x = tf.nn.bias_add(tf.matmul(fc11, fc2w1), fc2b1)
    fc2y = tf.nn.bias_add(tf.matmul(fc12, fc2w2), fc2b2)

    fc21 = activation(fc2x)
    fc22 = activation(fc2y)

    # fc3
    fc3w1 = tf.Variable(tf.truncated_normal(shape=[net_shape[1], net_shape[2]], dtype=tf.float32, stddev=1e-1))
    fc3w2 = tf.Variable(tf.truncated_normal(shape=[net_shape[1], net_shape[2]], dtype=tf.float32, stddev=1e-1))
    fc3b1 = tf.Variable(tf.constant(1e-1, shape=[net_shape[2]], dtype=tf.float32))
    fc3b2 = tf.Variable(tf.constant(1e-1, shape=[net_shape[2]], dtype=tf.float32))

    fc3x = tf.nn.bias_add(tf.matmul(fc21, fc3w1), fc3b1)
    fc3y = tf.nn.bias_add(tf.matmul(fc22, fc3w2), fc3b2)

    fc3x = activation(fc3x)
    fc3y = activation(fc3y)

    #fc3x - tf.cast(tf.divide(1, bands), tf.float32) * tf.matmul(fc3x, tf.ones([bands, bands]))
    m = tf.shape(fc3x)[1]
    fc_x = fc3x - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(fc3x, tf.ones([m, m]))
    fc_y = fc3y - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(fc3y, tf.ones([m, m]))

    Differ = fc_x - fc_y
    mean_x = tf.reduce_mean(fc_x, axis=0, keepdims=True)
    mean_y = tf.reduce_mean(fc_y, axis=0, keepdims=True)
    cov_xy = tf.divide(tf.matmul(tf.transpose(fc_x-mean_x ), fc_x- mean_y) ,
                       tf.cast(tf.shape(fc_x)[0], tf.float32))
                       
    A = tf.matmul(Differ, Differ, transpose_a=True)
    A = A / train_num

    sigmaX = tf.matmul(fc_x, fc_x, transpose_a=True)
    sigmaY = tf.matmul(fc_y, fc_y, transpose_a=True)
    sigmaX = sigmaX / train_num + args.reg  * tf.eye(net_shape[-1])#regularization
    sigmaY = sigmaY / train_num + args.reg  * tf.eye(net_shape[-1])

    B = (sigmaX + sigmaY) / 2# + args.reg * tf.eye(net_shape[-1])
    # # B_inv=tf.matrix_inverse (B)
    # # # B_inv, For numerical stability.
    D_B, V_B = tf.self_adjoint_eig(B)#tensor[...,:,:] * v[..., :,i] = e[..., i] * v[...,:,i],其中 i=0...N-1)
    # idx = tf.where(D_B > 1e-12)[:, 0]
    idx = tf.where(D_B > 1e-4)[:, 0]
    D_B = tf.gather(D_B, idx)
    V_B = tf.gather(V_B, idx, axis=1)#从params的axis维根据indices的参数值获取切片
    B_inv = tf.matmul(tf.matmul(V_B, tf.diag(tf.reciprocal(D_B))), tf.transpose(V_B))#reciprocal计算 x 元素的倒数.为什么为什么为什么
    # # B_inv = tf.matmul(tf.matmul(V_B, tf.diag(tf.reciprocal(D_B))), tf.matrix_inverse (V_B))
    sigma = tf.matmul(B_inv, A)#args.reg * tf.eye(net_shape[-1])

    D, V = tf.self_adjoint_eig(sigma)#计算B-1@A的特征值W
    
    #loss = tf.sqrt(tf.trace(tf.matmul(sigma,sigma)))
    loss = tf.trace(tf.matmul(sigma,sigma))

    optimizer = tf.train.GradientDescentOptimizer(args.lr).minimize(loss)

    init = tf.global_variables_initializer()

    loss_log = []

    gpu_options = tf.GPUOptions(allow_growth = True)
    conf        = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=conf)

    sess.run(init)
    #writer = tf.summary.FileWriter('graph')
    #writer.add_graph(sess.graph)
    i=0
    loss_log.append(9999)
    for k in range(args.epoch):
        _,cov_xy_,D_,idx_=sess.run([optimizer,cov_xy,D,idx], feed_dict={xd: xtrain, yd: ytrain})
        # print('idx_',idx_.shape,idx_)
        if k % 100 == 0:
            # print('cov_xy_',cov_xy_)
            # print('D_',D_)
            # print('D_B_',D_B_)
            ll = sess.run(loss, feed_dict={xd: xtrain, yd: ytrain})
            ll = ll / net_shape[-1]
            logging.info('The %4d-th epochs, loss is %4.4f ' % (k, ll))
            loss_log.append(ll)
            i=i+1
            if loss_log[i-1]-loss_log[i]<0.002:
                break
            
    matV = sess.run(V, feed_dict={xd: xtest, yd: ytest})
    bVal = sess.run(B, feed_dict={xd: xtest, yd: ytest})

    fcx = sess.run(fc_x, feed_dict={xd: xtest, yd: ytest})
    fcy = sess.run(fc_y, feed_dict={xd: xtest, yd: ytest})

    sess.close()
    # print('')

    return loss_log, matV, fcx, fcy, bVal


'''
采样样本总数(变化总数+未变化总数)： 111583 7080 104503
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[tn, fp, fn, tp]: 97584 1439 6919 5641
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OA: 0.9250961167919844 | kappa: 0.5368534230727537  | AUC: 0.7172961132011343
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
              precision    recall  f1-score   support

    unchange       0.93      0.99      0.96     99023
      change       0.80      0.45      0.57     12560

    accuracy                           0.93    111583
   macro avg       0.87      0.72      0.77    111583
weighted avg       0.92      0.93      0.92    111583

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




'''