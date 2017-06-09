import sys
sys.path.append('/Users/yuxpan/pylearn2/')

import numpy as np
import scipy as sp
import scipy.sparse
import pandas as pd

from pylearn2.models import mlp
from pylearn2.models.mlp import RectifiedLinear, MLP, Softmax
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets import DenseDesignMatrix
#from pylearn2.dataset.sparse_dataset import SparseDataset
from pylearn2.train import Train

from theano.compat.python2x import OrderedDict
import theano.tensor as T
from theano import function

import pickle, cPickle
import sklearn.preprocessing as pp
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.utils import shuffle

from datetime import datetime
import os, sys
from predict import predict
#import generateFeatures
from utility import *
from xgb_train_cvBodyId import *

def cv():

    #X2 = scaler.fit_transform(X ** .6)
    data_x, data_y, body_ids = build_data()
    
    holdout_ids = set([int(x.rstrip()) for x in file('hold_out_ids.txt')])
    print 'len(holdout_ids): ',len(holdout_ids)
    holdout_idx = [t for (t, x) in enumerate(body_ids) if x in holdout_ids]
    test_x = data_x[holdout_idx]
    print 'holdout_x.shape: '
    print test_x.shape
    test_y = data_y[holdout_idx]
    print Counter(test_y)
    
    #return 1

    cv_ids = set([int(x.rstrip()) for x in file('training_ids.txt')])
    print 'len(cv_ids): ',len(cv_ids)
    cv_idx = [t for (t, x) in enumerate(body_ids) if x in cv_ids]
    cv_x = data_x[cv_idx]
    print 'cv_x.shape: '
    print cv_x.shape
    cv_y = data_y[cv_idx]
    groups = body_ids[cv_idx] # GroupKFold will make sure all samples 
                              # having the same "Body ID" will appear in the same fold
    n_folds = 5
    scores = []
    best_iters = []
    kf = GroupKFold(n_splits=n_folds)
    
    for fold, (trainInd, validInd) in enumerate(kf.split(cv_x, cv_y, groups)):
        
        print 'fold %s' % fold
        x_train = cv_x[trainInd]
        y_train = cv_y[trainInd]
        x_valid = cv_x[validInd]
        y_valid = cv_y[validInd]
        idx_valid = np.array(cv_idx)[validInd]

        print 'y_train.shape:'
        print y_train.shape
        print 'x_train.shape:'
        print x_train.shape
        
        # standardize
        n_train = x_train.shape[0]
        train = np.vstack([x_train, x_valid])
        train = np.asarray(train, dtype=np.float64)
        #train = np.sign(train) * np.abs(train) ** 0.5
        scaler = pp.StandardScaler()
        train = scaler.fit_transform(train)
        
        x_train = train[:n_train,:]
        yMat = pd.get_dummies(y_train).values

        x_valid = train[n_train:,:]
        yMat_valid = pd.get_dummies(y_valid).values
        valid = DenseDesignMatrix(X = x_valid, y = yMat_valid)

        # [l1, l2, l3, l4, output]
        nIter = 1

        # Params for RI
        m = 256
        k = 64

        # Params for NN
        epochs = 30

        bs = 200
        mm = .1
        lr = .05
        dim2 = 512
        ir1 = .01
        ir2 = .05
        ip = .8
        ir_out = .05
        mcn_out = 2.5

        loglossAll = []
        best_iter = -1
        iepoch = 0
        best_logloss = 9999.
        t0 = datetime.now()
        for i in range(nIter):

            print 'iteration: ',i
            seed = i + 3819
            
            # shuffle training set for each nn model
            train = np.hstack((x_train, yMat))
            np.random.seed(seed)
            np.random.shuffle(train)
            x_train = train[:,:-3]
            yMat = train[:,-3:]
            print 'yMat.shape:'
            print yMat.shape
            print 'x_train.shape:'
            print x_train.shape
            #sys.exit(0)
            training = DenseDesignMatrix(X = x_train, y = yMat)

            R = RImatrix(x_train.shape[1], m, k, rm_dup_cols = True, seed = seed)
            R = np.abs(R.todense().astype(np.float32))
            dim1 = R.shape[1]
            l1 = RectifiedLinear(layer_name='l1', irange = ir1, dim = dim1, mask_weights = R)
            l2 = RectifiedLinear(layer_name='l2', irange = ir2, dim = dim2, max_col_norm = 1.)
            l3 = RectifiedLinear(layer_name='l3', irange = ir2, dim = dim2, max_col_norm = 1.)
            #l4 = RectifiedLinear(layer_name='l4', irange = ir2, dim = dim2, max_col_norm = 1.)
            output = Softmax(layer_name='y', n_classes = 4, irange = ir2, max_col_norm = mcn_out)
            #mdl = MLP([l1, l2, l3, l4, output], nvis = X2.shape[1])
            mdl = MLP([l1, l2, l3, output], nvis = x_train.shape[1])
            trainer = sgd.SGD(learning_rate=lr,
                              batch_size=bs,
                              learning_rule=learning_rule.Momentum(mm),
                              cost=Dropout(input_include_probs = {'l1':1.},
                                           input_scales = {'l1':1.},
                                           default_input_include_prob=ip,
                                           default_input_scale=1/ip),
                              termination_criterion=EpochCounter(epochs),
                              seed=seed)
                              #monitoring_dataset=valid, seed=seed)
            decay = sgd.LinearDecayOverEpoch(start=2, saturate=30, decay_factor= .1)
            trainer.setup(mdl, training)
            while True:
                trainer.train(dataset=training)
                mdl.monitor.report_epoch()
                mdl.monitor()
                
                if not trainer.continue_learning(mdl):
                    break
                decay.on_monitor(mdl, valid, trainer)

                compute_logloss = True
                if compute_logloss:
                    iepoch += 1
                    #pred_valid = predict(mdl, x_valid.astype(np.float32)).reshape(x_valid.shape[0], 3) 
                    pred_valid = predict(mdl, x_valid.astype(np.float32))
                    #print 'pred_valid'
                    #print pred_valid
                    logloss = log_loss(y_valid, pred_valid, labels = [0.0, 1.0, 2.0, 3.0])
                    output = '------------> logloss: %f' % logloss
                    loglossAll.append(logloss)
                    if logloss < best_logloss:
                        best_iter = iepoch
                        best_logloss = logloss
                    print output
        
        for i in xrange(len(loglossAll)):
            print 'iter %d, logloss: %f' % (i+1, loglossAll[i])

        print 'best iteration: %d, best logloss: %f' % (best_iter, best_logloss)
        scores.append(best_logloss)
        best_iters.append(best_iter)
        # save y_valid and valid_pred to file
        #pred_valid = predict(mdl, x_valid.astype(np.float32)).reshape(x_valid.shape[0], 3)
        #pred_valid = predict(mdl, x_valid.astype(np.float32))
        #valid_pred = pred_valid
        #with open('nn_exp4.pkl', 'wb') as outfile:
        #    cPickle.dump(feat_names, outfile, -1)
        #    cPickle.dump(valid_ids, outfile, -1)
        #    cPickle.dump(valid_pred, outfile, -1)
        #    cPickle.dump(y_valid, outfile, -1)
        #print 'valid_pred saved'

        #break

    #print 'scores:'
    #print scores
    #print 'mean score:'
    #print np.mean(scores)
    #print 'best iterations:'
    #print best_iters
    #print 'mean iters:'
    #print np.mean(best_iters)


if __name__ == '__main__':
    cv()

