from FeatureGenerator import *
import pandas as pd
import numpy as np
import cPickle
import gensim
from sklearn.preprocessing import normalize
from helpers import *


class Word2VecFeatureGenerator(FeatureGenerator):


    def __init__(self, name='word2vecFeatureGenerator'):
        super(Word2VecFeatureGenerator, self).__init__(name)


    def process(self, df):

        print 'generating word2vec features'
        df["Headline_unigram_vec"] = df["Headline"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))
        df["articleBody_unigram_vec"] = df["articleBody"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))
        
        n_train = df[~df['target'].isnull()].shape[0]
        print 'Word2VecFeatureGenerator: n_train:',n_train
        n_test = df[df['target'].isnull()].shape[0]
        print 'Word2VecFeatureGenerator: n_test:',n_test
        
        # 1). document vector built by multiplying together all the word vectors
        # using Google's pre-trained word vectors
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        print 'model loaded'

        Headline_unigram_array = df['Headline_unigram_vec'].values
        print 'Headline_unigram_array:'
        print Headline_unigram_array
        print Headline_unigram_array.shape
        print type(Headline_unigram_array)
        
        # word vectors weighted by normalized tf-idf coefficient?
        #headlineVec = [0]
        headlineVec = map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*300), Headline_unigram_array)
        headlineVec = np.array(headlineVec)
        print 'headlineVec:'
        print headlineVec
        print 'type(headlineVec)'
        print type(headlineVec)
        #headlineVec = np.exp(headlineVec)
        headlineVec = normalize(headlineVec)
        print 'headlineVec'
        print headlineVec
        print headlineVec.shape
        
        headlineVecTrain = headlineVec[:n_train, :]
        outfilename_hvec_train = "train.headline.word2vec.pkl"
        with open(outfilename_hvec_train, "wb") as outfile:
            cPickle.dump(headlineVecTrain, outfile, -1)
        print 'headline word2vec features of training set saved in %s' % outfilename_hvec_train

        if n_test > 0:
            # test set is available
            headlineVecTest = headlineVec[n_train:, :]
            outfilename_hvec_test = "test.headline.word2vec.pkl"
            with open(outfilename_hvec_test, "wb") as outfile:
                cPickle.dump(headlineVecTest, outfile, -1)
            print 'headline word2vec features of test set saved in %s' % outfilename_hvec_test
        print 'headine done'

        Body_unigram_array = df['articleBody_unigram_vec'].values
        print 'Body_unigram_array:'
        print Body_unigram_array
        print Body_unigram_array.shape
        #bodyVec = [0]
        bodyVec = map(lambda x: reduce(np.add, [model[y] for y in x if y in model], [0.]*300), Body_unigram_array)
        bodyVec = np.array(bodyVec)
        bodyVec = normalize(bodyVec)
        print 'bodyVec'
        print bodyVec
        print bodyVec.shape

        bodyVecTrain = bodyVec[:n_train, :]
        outfilename_bvec_train = "train.body.word2vec.pkl"
        with open(outfilename_bvec_train, "wb") as outfile:
            cPickle.dump(bodyVecTrain, outfile, -1)
        print 'body word2vec features of training set saved in %s' % outfilename_bvec_train
        
        if n_test > 0:
            # test set is available
            bodyVecTest = bodyVec[n_train:, :]
            outfilename_bvec_test = "test.body.word2vec.pkl"
            with open(outfilename_bvec_test, "wb") as outfile:
                cPickle.dump(bodyVecTest, outfile, -1)
            print 'body word2vec features of test set saved in %s' % outfilename_bvec_test

        print 'body done'

        # compute cosine similarity between headline/body word2vec features
        simVec = np.asarray(map(cosine_sim, headlineVec, bodyVec))[:, np.newaxis]
        print 'simVec.shape:'
        print simVec.shape

        simVecTrain = simVec[:n_train]
        outfilename_simvec_train = "train.sim.word2vec.pkl"
        with open(outfilename_simvec_train, "wb") as outfile:
            cPickle.dump(simVecTrain, outfile, -1)
        print 'word2vec sim. features of training set saved in %s' % outfilename_simvec_train
        
        if n_test > 0:
            # test set is available
            simVecTest = simVec[n_train:]
            outfilename_simvec_test = "test.sim.word2vec.pkl"
            with open(outfilename_simvec_test, "wb") as outfile:
                cPickle.dump(simVecTest, outfile, -1)
            print 'word2vec sim. features of test set saved in %s' % outfilename_simvec_test

        return 1

    def read(self, header='train'):

        filename_hvec = "%s.headline.word2vec.pkl" % header
        with open(filename_hvec, "rb") as infile:
            headlineVec = cPickle.load(infile)

        filename_bvec = "%s.body.word2vec.pkl" % header
        with open(filename_bvec, "rb") as infile:
            bodyVec = cPickle.load(infile)

        filename_simvec = "%s.sim.word2vec.pkl" % header
        with open(filename_simvec, "rb") as infile:
            simVec = cPickle.load(infile)

        print 'headlineVec.shape:'
        print headlineVec.shape
        #print type(headlineVec)
        print 'bodyVec.shape:'
        print bodyVec.shape
        #print type(bodyVec)
        print 'simVec.shape:'
        print simVec.shape
        #print type(simVec)

        return [headlineVec, bodyVec, simVec]
        #return [simVec.reshape(-1,1)]

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
