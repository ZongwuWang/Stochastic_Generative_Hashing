
import time
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import function

import scipy.io as sio
import sys
import pickle
import argparse


"""This is a simple demonstration of the stochastic generative hashing algorithm 
with linear decoder and encoder on MNIST dataset. 

Created by Bo Dai 2016"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function

import scipy.io as sio
import sys

def VAE_stoc_neuron(alpha, dim_input, dim_hidden, batch_size, learning_rate, max_iter, xtrain, xvar, xmean):
    
    g = tf.Graph()
    dtype = tf.float32
    
    with g.as_default():
        # x = tf.placeholder(dtype, [None, dim_input])
        x = tf.compat.v1.placeholder(dtype, [None, dim_input])
        
        # define doubly stochastic neuron with gradient by DeFun
        @function.Defun(dtype, dtype, dtype, dtype)
        def DoublySNGrad(logits, epsilon, dprev, dpout):
            prob = 1.0 / (1 + tf.exp(-logits))
            yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
            # {-1, 1} coding
            # yout = tf.sign(prob - epsilon)

            # biased
            dlogits = prob * (1 - prob) * (dprev + dpout)
                        
            depsilon = dprev
            return dlogits, depsilon

        @function.Defun(dtype, dtype, grad_func=DoublySNGrad)
        def DoublySN(logits, epsilon):
            prob = 1.0 / (1 + tf.exp(-logits))
            yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
            return yout, prob

        '''
        def compute_expression(x, z):
            zero = tf.constant(0.0, dtype=tf.float32)
            one = tf.constant(1.0, dtype=tf.float32)
            abs_x = tf.abs(x)
            max_x = tf.maximum(x, zero)
            exp_term = tf.exp(-abs_x)
            log_term = tf.math.log(one + exp_term)

            result = max_x - (x * z) + log_term
            return result
        '''

        with tf.name_scope('encode'):
            wencode = tf.Variable(tf.random.normal([dim_input, dim_hidden], stddev=1.0 / tf.sqrt(float(dim_input)), dtype=dtype),
                                       name='wencode')
            bencode = tf.Variable(tf.random.normal([dim_hidden], dtype=dtype), name='bencode')
            hencode = tf.matmul(x, wencode) + bencode
            # determinastic output
            hepsilon = tf.ones(shape=tf.shape(hencode), dtype=dtype) * .5
            
        yout, pout = DoublySN(hencode, hepsilon)
        
        with tf.name_scope('decode'):
            wdecode = tf.Variable(tf.random.normal([dim_hidden, dim_input], stddev=1.0 / tf.sqrt(float(dim_hidden)), dtype=dtype), 
                                  name='wdecode')
        with tf.name_scope('scale'):
            scale_para = tf.Variable(tf.constant(xvar, dtype=dtype), name="scale_para")
            shift_para = tf.Variable(tf.constant(xmean, dtype=dtype), name="shift_para")
            
        xout = tf.matmul(yout, wdecode) * tf.abs(scale_para) + shift_para
        
        monitor = tf.nn.l2_loss(xout - x, name=None) 
        # monitor = tf.reduce_sum(tf.abs(xout - x), name=None)
        # loss = monitor + alpha * tf.reduce_sum(tf.reduce_sum(yout * tf.math.log(pout) + (1 - yout) * tf.math.log(1 - pout))) + beta * tf.nn.l2_loss(wdecode, name=None)
        alpha_loss = alpha * tf.reduce_sum(tf.reduce_sum(yout * tf.math.log(pout) + (1 - yout) * tf.math.log(1 - pout)))
        beta_loss = beta * tf.nn.l2_loss(wdecode, name=None)
        # loss = monitor + alpha_loss + beta_loss
        loss = monitor

        # alpha_loss = alpha * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(hencode, yout)) 
        # beta_loss = beta * tf.nn.l2_loss(wdecode, name=None) + beta * tf.nn.l2_loss(wencode, name=None)
        # loss = monitor + alpha_loss + beta_loss
        # loss = monitor
        # beta_loss = tf.zeros_like(beta_loss)
        # loss = monitor + alpha * alpha_loss + beta * beta_loss
        
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        # optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)
        # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

        # sess = tf.Session(graph=g)
        sess = tf.compat.v1.Session(graph=g)
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.compat.v1.global_variables_initializer())
        
        train_err = []
        # Initialize the best loss value to a large value
        best_loss = float('inf')
        best_results = None
        for i in range(max_iter):
            indx = np.random.choice(xtrain.shape[0], batch_size)
            xbatch = xtrain[indx]
            _, monitor_value, loss_value, alpha_loss_value, beta_loss_value = sess.run([train_op, monitor, loss, alpha_loss, beta_loss], feed_dict={x: xbatch})
            # Check if the current loss is better than the best loss so far
            if loss_value < best_loss:
                best_loss = loss_value
                # Save the best results (model parameters and other relevant variables)
                node_list = ['yout', 'pout', 'xout', 'wencode', 'bencode', 'wdecode', 'scale_para', 'shift_para']
                t_vars = tf.compat.v1.trainable_variables()

                para_list = {}
                for var in t_vars:
                    para_list[var.name] = sess.run(var)

                best_results = {
                    'node_list': node_list,
                    'para_list': para_list,
                    'train_err': train_err.copy()
                }

            if i % 100 == 0:
                # print('Num iteration: %d Loss: %0.04f; Monitor Loss %0.04f; Alpha Loss %0.04f; Beta Loss %0.04f' % (i, loss_value / batch_size, monitor_value / batch_size, alpha_loss_value / batch_size, beta_loss_value / batch_size))
                print('Num iteration: %d Loss: %0.04f; Monitor Loss %0.04f; Alpha Loss %0.04f; Beta Loss %0.04f' % (i, loss_value, monitor_value, alpha_loss_value, beta_loss_value))
                train_err.append(loss_value)

            if i % 1000 == 0:
                learning_rate = 0.1 * learning_rate


        # node_list = ['yout', 'pout', 'xout', 'wencode', 'bencode', 'wdecode', 'scale_para', 'shift_para']
        # # t_vars = tf.trainable_variables()
        # t_vars = tf.compat.v1.trainable_variables()

        # para_list = {}
        # for var in t_vars:
        #     para_list[var.name] = sess.run(var)
    
    # return g, node_list, para_list, train_err
    # Return the best results at the end
    print("Best loss = ", best_loss)
    return g, best_results['node_list'], best_results['para_list'], best_results['train_err']

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--dim_exp', type=int, default=4, help='binary dimension expension factor to the input dimension')
    argparser.add_argument('-b', '--batch_size', type=int, default=1024, help='batch size')
    argparser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('-e', '--max_iter', type=int, default=10000, help='max iteration')
    argparser.add_argument('--alpha', type=float, default=1e-4, help='alpha')
    argparser.add_argument('--beta', type=float, default=1e-3, help='beta')

    args = argparser.parse_args()

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.disable_eager_execution()
    # prepare data
    # please replace the dataset with your own directory.
    # traindata = sio.loadmat('../dataset/mnist_training.mat')
    # testdata = sio.loadmat('../dataset/mnist_test.mat')
    # xtrain = traindata['Xtraining']
    # xtest = testdata['Xtest']
    # xtrain_org = xtrain
    # xtrain = np.random.rand(60000, 784)
    # xtest = np.random.rand(60000, 784)
    # 生成高斯分布
    # xtrain = np.random.normal(0, 1, (60000, 784))
    # xtest = np.random.normal(0, 1, (60000, 784))
    # xtrain = np.load('../dataset/item_embs.npy')
    # xtest = np.load('../dataset/user_embs.npy')
    # xtrain = np.load('/data/wzw/AI/DLRM_Acce/DeepMatch/examples/item_embs.npy')
    # xtest = np.load('/data/wzw/AI/DLRM_Acce/DeepMatch/examples/user_embs.npy')
    # xtrain = pickle.load(open('/data/wzw/AI/HPCA24/torch-rechub/examples/matching/data/ml-20m/saved/cold_movie_embedding_weight.pkl', 'rb')).astype('float64')
    xtrain = pickle.load(open('../dataset/cold_movie_embedding_weight.pkl', 'rb')).astype('float64')

    # algorithm parameters
    # dim_input = 28 * 28
    dim_input = xtrain.shape[1]
    
    # normilize xtrain and xtest
    # xtrain_max = xtrain.max(axis=0).max().astype('float64')
    # xtrain_min = xtrain.min(axis=0).max().astype('float64')

    # xtest_max = xtest.max(axis=0).astype('float64')
    # xtest_min = xtest.min(axis=0).astype('float64')
    # print(xtrain[:10, :].astype('float64'))
    # xtrain_org = xtrain
    # xtrain = 1 / (1 + np.exp(-k * (xtrain - midpoint)))
    # xtrain = (xtrain - xtrain_min) / (xtrain_max - xtrain_min)
    # xtest = (xtest - xtest_min) / (xtest_max - xtest_min)
    # print(xtrain[:10])
    '''
    print("max xtrain = ", xtrain.flatten().max())
    print("min xtrain = ", xtrain.flatten().min())
    xtrain = (xtrain - xtrain.flatten().min()) / (xtrain.flatten().max() - xtrain.flatten().min())
    xtrain = np.log2(xtrain + 2e-1) + 1
    '''
    xtrain = (xtrain - xtrain.flatten().min()) / (xtrain.flatten().max() - xtrain.flatten().min())
    # xtrain = (xtrain - xtrain.min()) / (xtrain.max() - xtrain.min())
    print("mean xtrain = ", xtrain.mean())
    print("var xtrain = ", xtrain.var())

    xmean = xtrain.mean(axis=0).astype('float64')
    xvar = np.clip(xtrain.var(axis=0), 1e-7, np.inf).astype('float64')
    # print("max xtrain = ", xtrain.max())
    # print("min xtrain = ", xtrain.min())
    # print("mean xtrain = ", xtrain.mean())
    # print("var xtrain = ", xtrain.var())
    # print("max xtest = ", xtest.max())
    # print("min xtest = ", xtest.min())
    # print("mean xtest = ", xtest.mean())
    # print("var xtest = ", xtest.var())
    # import os
    # os.kill()

    # length of bits
    # dim_hidden= int(sys.argv[1]) 
    dim_hidden = dim_input * args.dim_exp
    print('dim of hidden variable is %d' %(dim_hidden))

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    max_iter = args.max_iter
    alpha = args.alpha
    beta = args.beta

    # start training
    start_time = time.time()
    g, node_list, para_list, train_err = VAE_stoc_neuron(alpha, dim_input, dim_hidden, batch_size, learning_rate, max_iter, xtrain, xvar, xmean)
    end_time = (time.time() - start_time)

    print('Running time: %0.04f s' %end_time)

    W = para_list['encode/wencode:0']
    b = para_list['encode/bencode:0']
    U = para_list['decode/wdecode:0']
    shift = para_list['scale/shift_para:0']
    scale = para_list['scale/scale_para:0']

    trainlogits = np.dot(np.array(xtrain), W) + b
    # epsilon = np.random.uniform(0, 1, logits.shape)
    epsilon = 0.5

    trainpres = 1.0 / (1 + np.exp(-trainlogits))
    htrain = (np.sign(trainpres - epsilon) + 1.0) / 2.0
    print(xtrain[:5, :].astype('float64'))
    print(htrain[:5, :])
    hconstruct = np.matmul(htrain, U) * np.abs(scale) + shift
    print(hconstruct[:5, :])
    l2_dist = np.sum(np.sqrt(np.sum(np.square(hconstruct - xtrain), axis=1)))
    print("l2_dist = ", l2_dist)

    # hconstruct = hconstruct * (xtrain_max - xtrain_min) + xtrain_min
    # print(hconstruct[:10, :])
    # # 计算hconstruct和xtrain的l2距离
    # l2_dist = np.sum(np.sqrt(np.sum(np.square(hconstruct - xtrain_org), axis=1)))
    # print("l2_dist = ", l2_dist)