import pandas as pd
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import log_loss
import cPickle
from score import *
from xgb_train_cvBodyId import fscore, perfect_score

def load_data():
    
    yuxi = pd.read_csv('predtest_cor2.csv', usecols=['Headline','Body ID','Stance','prob_0','prob_1','prob_2','prob_3'])
    print 'yuxi.shape:'
    print yuxi.shape
    

    doug = pd.read_csv('dosiblOutput.csv', usecols=['Headline','Body ID','Agree','Disagree','Discuss','Unrelated'])
    print 'doug.shape:'
    print doug.shape
    combine = pd.merge(yuxi, doug, on=['Headline', 'Body ID'], how='inner')
    print 'combine.shape:'
    print combine.shape

    targets = ['agree', 'disagree', 'discuss', 'unrelated']
    targets_dict = dict(zip(targets, range(len(targets))))
    combine['target'] = map(lambda x: targets_dict[x], combine['Stance'])
    y_meta = combine['target'].values
    x_meta = combine[['prob_0','prob_1','prob_2','prob_3','Agree','Disagree','Discuss','Unrelated']].values
    
    return x_meta, y_meta


def loadTest():
    
    yuxi = pd.read_csv('tree_pred_prob_cor2.csv', usecols=['Headline','Body ID', 'prob_0','prob_1','prob_2','prob_3'])
    print 'yuxi.shape:'
    print yuxi.shape

    doug = pd.read_csv('dosiblOutputFinal.csv', usecols=['Headline','Body ID','Agree','Disagree','Discuss','Unrelated'])  
    print 'doug.shape:'
    print doug.shape

    combine = pd.concat([yuxi, doug], axis=1)
    print 'combine.shape:'
    print combine.shape
    
    x_meta = combine[['prob_0','prob_1','prob_2','prob_3','Agree','Disagree','Discuss','Unrelated']].values

    return x_meta

def stack_test():
    
    param = {
        'w0': 1.0,
        'w1': 1.0
    }
    sumw = param['w0'] + param['w1']
    x_meta = loadTest()
    
    pred_agree = (x_meta[:,0]*param['w0'] + x_meta[:,4]*param['w1']) / sumw
    pred_disagree = (x_meta[:,1]*param['w0'] + x_meta[:,5]*param['w1']) / sumw
    pred_discuss = (x_meta[:,2]*param['w0'] + x_meta[:,6]*param['w1']) / sumw
    pred_unrelated = (x_meta[:,3]*param['w0'] + x_meta[:,7]*param['w1']) / sumw

    pred_y = np.hstack([pred_agree.reshape((-1,1)), pred_disagree.reshape((-1,1)), pred_discuss.reshape((-1,1)), pred_unrelated.reshape((-1,1))])
    print 'pred_agree.shape:'
    print pred_agree.shape
    print 'pred_disagree.shape:'
    print pred_disagree.shape
    print 'pred_discuss.shape:'
    print pred_discuss.shape
    print 'pred_unrelated.shape:'
    print pred_unrelated.shape

    print 'pred_y.shape:'
    print pred_y.shape
    pred_y_idx = np.argmax(pred_y, axis=1)
    predicted = [LABELS[int(a)] for a in pred_y_idx]
    
    stances = pd.read_csv("test_stances_unlabeled_processed.csv")
    df_output = pd.DataFrame()
    df_output['Headline'] = stances['Headline']
    df_output['Body ID'] = stances['Body ID']
    df_output['Stance'] = predicted
    df_output.to_csv('averaged_2models_cor4.csv', index=False)


def stack_cv(param):
    
    #x_meta, y_meta = load_data()
    sumw = param['w0'] + param['w1'] 
    pred_agree = (x_meta[:,0]*param['w0'] + x_meta[:,4]*param['w1']) / sumw
    pred_disagree = (x_meta[:,1]*param['w0'] + x_meta[:,5]*param['w1']) / sumw
    pred_discuss = (x_meta[:,2]*param['w0'] + x_meta[:,6]*param['w1']) / sumw
    pred_unrelated = (x_meta[:,3]*param['w0'] + x_meta[:,7]*param['w1']) / sumw

    pred_y = np.hstack([pred_agree.reshape((-1,1)), pred_disagree.reshape((-1,1)), pred_discuss.reshape((-1,1)), pred_unrelated.reshape((-1,1))])
    print 'pred_agree.shape:'
    print pred_agree.shape
    print 'pred_disagree.shape:'
    print pred_disagree.shape
    print 'pred_discuss.shape:'
    print pred_discuss.shape
    print 'pred_unrelated.shape:'
    print pred_unrelated.shape

    print 'pred_y.shape:'
    print pred_y.shape
    print 'y_meta.shape:'
    print y_meta.shape
    
    pred_y_label = np.argmax(pred_y, axis=1)
    predicted = [LABELS[int(a)] for a in pred_y_label]
    actual = [LABELS[int(a)] for a in y_meta]    

    score, _ = score_submission(actual, predicted)
    s_perf, _ = score_submission(actual, actual)

    cost = float(score) / s_perf

    #cost = log_loss(y_meta, pred_y, labels = [0, 1, 2, 3])
    
    return -1.0 * cost


def hyperopt_wrapper(param):
    
    print "++++++++++++++++++++++++++++++"
    for k, v in sorted(param.items()):
        print "%s: %s" % (k,v)

    loss = stack_cv(param)
    print "-cost: ", loss

    return {'loss': loss, 'status': STATUS_OK}

def run():

    param_space = {

            'w0': 1.0,
            'w1': hp.quniform('w1', 0.01, 2.0, 0.01),
            'max_evals': 800
            }
    
    
    trial_counter = 0
    trials = Trials()
    objective = lambda p: hyperopt_wrapper(p)
    best_params = fmin(objective, param_space, algo=tpe.suggest,\
        trials = trials, max_evals=param_space["max_evals"])
    
    print 'best parameters: '
    for k, v in best_params.items():
        print "%s: %s" % (k ,v)
    
    trial_loss = np.asarray(trials.losses(), dtype=float)
    best_loss = min(trial_loss)
    print 'best loss: ', best_loss


if __name__ == '__main__':
    
    #x_meta, y_meta = load_data()
    #run()
    #loadTest()
    stack_test()
    #stack_test()

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
