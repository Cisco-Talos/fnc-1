from FeatureGenerator import *
import pandas as pd
import numpy as np
import cPickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from helpers import *


class SentimentFeatureGenerator(FeatureGenerator):


    def __init__(self, name='sentimentFeatureGenerator'):
        super(SentimentFeatureGenerator, self).__init__(name)


    def process(self, df):

        print 'generating sentiment features'
        print 'for headline'
        
        n_train = df[~df['target'].isnull()].shape[0]
        n_test = df[df['target'].isnull()].shape[0]

        # calculate the polarity score of each sentence then take the average
        sid = SentimentIntensityAnalyzer()
        def compute_sentiment(sentences):
            result = []
            for sentence in sentences:
                vs = sid.polarity_scores(sentence)
                result.append(vs)
            return pd.DataFrame(result).mean()
        
        #df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x.decode('utf-8')))
        df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['headline_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'h_compound', 'neg':'h_neg', 'neu':'h_neu', 'pos':'h_pos'}, inplace=True)
        #print 'df:'
        #print df
        #print df.columns
        #print df.shape
        headlineSenti = df[['h_compound','h_neg','h_neu','h_pos']].values
        print 'headlineSenti.shape:'
        print headlineSenti.shape
        
        headlineSentiTrain = headlineSenti[:n_train, :]
        outfilename_hsenti_train = "train.headline.senti.pkl"
        with open(outfilename_hsenti_train, "wb") as outfile:
            cPickle.dump(headlineSentiTrain, outfile, -1)
        print 'headline sentiment features of training set saved in %s' % outfilename_hsenti_train
        
        if n_test > 0:
            # test set is available
            headlineSentiTest = headlineSenti[n_train:, :]
            outfilename_hsenti_test = "test.headline.senti.pkl"
            with open(outfilename_hsenti_test, "wb") as outfile:
                cPickle.dump(headlineSentiTest, outfile, -1)
            print 'headline sentiment features of test set saved in %s' % outfilename_hsenti_test
        
        print 'headine senti done'
        
        #return 1

        print 'for body'
        #df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x.decode('utf-8')))
        df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['body_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'b_compound', 'neg':'b_neg', 'neu':'b_neu', 'pos':'b_pos'}, inplace=True)
        #print 'body df:'
        #print df
        #print df.columns
        bodySenti = df[['b_compound','b_neg','b_neu','b_pos']].values
        print 'bodySenti.shape:'
        print bodySenti.shape
        
        bodySentiTrain = bodySenti[:n_train, :]
        outfilename_bsenti_train = "train.body.senti.pkl"
        with open(outfilename_bsenti_train, "wb") as outfile:
            cPickle.dump(bodySentiTrain, outfile, -1)
        print 'body sentiment features of training set saved in %s' % outfilename_bsenti_train
        
        if n_test > 0:
            # test set is available
            bodySentiTest = bodySenti[n_train:, :]
            outfilename_bsenti_test = "test.body.senti.pkl"
            with open(outfilename_bsenti_test, "wb") as outfile:
                cPickle.dump(bodySentiTest, outfile, -1)
            print 'body sentiment features of test set saved in %s' % outfilename_bsenti_test

        print 'body senti done'

        return 1


    def read(self, header='train'):

        filename_hsenti = "%s.headline.senti.pkl" % header
        with open(filename_hsenti, "rb") as infile:
            headlineSenti = cPickle.load(infile)

        filename_bsenti = "%s.body.senti.pkl" % header
        with open(filename_bsenti, "rb") as infile:
            bodySenti = cPickle.load(infile)

        print 'headlineSenti.shape:'
        print headlineSenti.shape
        #print type(headlineSenti)
        print 'bodySenti.shape:'
        print bodySenti.shape
        #print type(bodySenti)

        return [headlineSenti, bodySenti]

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
