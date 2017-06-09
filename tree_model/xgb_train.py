#!/usr/bin/env python

import sys
import cPickle
import numpy as np
from itertools import chain
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from collections import Counter
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *

params_xgb = {
    'max_depth': 6,
    'colsample_bytree': 0.6,
    'subsample': 1.0,
    'eta': 0.1,
    'silent': 1,
    'objective': 'multi:softmax',
    'eval_metric':'mlogloss',
    'num_class': 4
}
num_round = 1000

def build_data():
    
    # create target variable
    body = pd.read_csv("train_bodies.csv")
    stances = pd.read_csv("train_stances.csv")
    data = pd.merge(body, stances, how='right', on='Body ID')
    targets = ['agree', 'disagree', 'discuss', 'unrelated']
    targets_dict = dict(zip(targets, range(len(targets))))
    data['target'] = map(lambda x: targets_dict[x], data['Stance'])
    
    data_y = data['target'].values
    
    # read features
    generators = [
                  CountFeatureGenerator(),
                  TfidfFeatureGenerator(),
                  SvdFeatureGenerator(),
                  Word2VecFeatureGenerator(),
                  SentimentFeatureGenerator()
                 ]

    features = [f for g in generators for f in g.read()]

    data_x = np.hstack(features)

    print 'data_x.shape'
    print data_x.shape
    print 'data_y.shape'
    print data_y.shape

    return data_x, data_y

def fscore(pred_y, truth_y):
    
    # targets = ['agree', 'disagree', 'discuss', 'unrelated']
    # y = [0, 1, 2, 3]
    score = 0
    if pred_y.shape != truth_y.shape:
        raise Exception('pred_y and truth have different shapes')
    for i in range(pred_y.shape[0]):
        if truth_y[i] == 3:
            if pred_y[i] == 3: score += 0.25
        else:
            if pred_y[i] != 3: score += 0.25
            if truth_y[i] == pred_y[i]: score += 0.75
    
    return score

def perfect_score(truth_y):
    
    score = 0
    for i in range(truth_y.shape[0]):
        if truth_y[i] == 3: score += 0.25
        else: score += 1
        #else: score += 0.75

    return score

def cv():
    
    data_x, data_y = build_data()
    
    random_seed = 2017
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    #with open('skf.pkl', 'wb') as outfile:
    #    cPickle.dump(skf, outfile, -1)
    #    print 'skf saved'
    
    scores = []
    best_iters = [0]*5
    pscores = []
    with open('skf.pkl', 'rb') as infile:
        skf = cPickle.load(infile)

        for fold, (trainInd, validInd) in enumerate(skf.split(data_x, data_y)):
            print 'fold %s' % fold
            x_train = data_x[trainInd]
            y_train = data_y[trainInd]
            x_valid = data_x[validInd]
            y_valid = data_y[validInd]
            
            print 'perfect_score: ', perfect_score(y_valid)
            print Counter(y_valid)
            #break
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dvalid = xgb.DMatrix(x_valid, label=y_valid)
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            bst = xgb.train(params_xgb, 
                            dtrain,
                            num_round,
                            watchlist,
                            verbose_eval=100)
                            #early_stopping_rounds=30)
            #pred_y = bst.predict(dvalid, ntree_limit=bst.best_ntree_limit)
            #print 'best iterations: ', bst.best_ntree_limit
            pred_y = bst.predict(dvalid)
            print pred_y
            print Counter(pred_y)
            #pred_y = np.argmax(bst.predict(dvalid, ntree_limit=bst.best_ntree_limit), axis=1)
            print 'pred_y.shape'
            print pred_y.shape
            print 'y_valid.shape'
            print y_valid.shape
            s = fscore(pred_y, y_valid)
            s_perf = perfect_score(y_valid)
            print 'fold %s, score = %d, perfect_score %d' % (fold, s, s_perf)
            scores.append(s)
            pscores.append(s_perf)
            #break

    print 'scores:'
    print scores
    print 'mean score:'
    print np.mean(scores)
    print 'perfect scores:'
    print pscores
    print 'mean perfect score:'
    print np.mean(pscores)

if __name__ == '__main__':
    #build_data()
    cv()

 #   Copyright 2017 Cisco Systems, Inc.
 #  
 #   Licensed under the Apache License, Version 2.0 (the "License");
 #   you may not use this file except in compliance with the License.
 #   You may obtain a copy of the License at
 #  
 #     http://www.apache.org/licenses/LICENSE-2.0
 #  
 #   Unless required by applicable law or agreed to in writing, software
 #   distributed under the License is distributed on an "AS IS" BASIS,
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #   See the License for the specific language governing permissions and
 #   limitations under the License.
