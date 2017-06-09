from FeatureGenerator import *
import ngram
import cPickle
import pandas as pd
from nltk.tokenize import sent_tokenize
from helpers import *
import hashlib


class CountFeatureGenerator(FeatureGenerator):


    def __init__(self, name='countFeatureGenerator'):
        super(CountFeatureGenerator, self).__init__(name)


    def process(self, df):

        grams = ["unigram", "bigram", "trigram"]
        feat_names = ["Headline", "articleBody"]
        print "generate counting features"
        for feat_name in feat_names:
            for gram in grams:
                df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
                df["count_of_unique_%s_%s" % (feat_name, gram)] = \
		            list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
                df["ratio_of_unique_%s_%s" % (feat_name, gram)] = \
                    map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)])

        # overlapping n-grams count
        for gram in grams:
            df["count_of_Headline_%s_in_articleBody" % gram] = \
                list(df.apply(lambda x: sum([1. for w in x["Headline_" + gram] if w in set(x["articleBody_" + gram])]), axis=1))
            df["ratio_of_Headline_%s_in_articleBody" % gram] = \
                map(try_divide, df["count_of_Headline_%s_in_articleBody" % gram], df["count_of_Headline_%s" % gram])
        
        # number of sentences in headline and body
        for feat_name in feat_names:
            #df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x.decode('utf-8').encode('ascii', errors='ignore'))))
            df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x)))
            #print df['len_sent_%s' % feat_name]

        # dump the basic counting features into a file
        feat_names = [ n for n in df.columns \
                if "count" in n \
                or "ratio" in n \
                or "len_sent" in n]
        
        # binary refuting features
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        _hedging_seed_words = [
            'alleged', 'allegedly',
            'apparently',
            'appear', 'appears',
            'claim', 'claims',
            'could',
            'evidently',
            'largely',
            'likely',
            'mainly',
            'may', 'maybe', 'might',
            'mostly',
            'perhaps',
            'presumably',
            'probably',
            'purported', 'purportedly',
            'reported', 'reportedly',
            'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
            'says',
            'seem',
            'somewhat',
            # 'supposedly',
            'unconfirmed'
        ]
        
        #df['refuting_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #df['hedging_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #check_words = _refuting_words + _hedging_seed_words
        check_words = _refuting_words
        for rf in check_words:
            fname = '%s_exist' % rf
            feat_names.append(fname)
            df[fname] = df['Headline'].map(lambda x: 1 if rf in x else 0)
	    
        # number of body texts paired up with the same headline
        #df['headline_hash'] = df['Headline'].map(lambda x: hashlib.md5(x).hexdigest())
        #nb_dict = df.groupby(['headline_hash'])['Body ID'].nunique().to_dict()
        #df['n_bodies'] = df['headline_hash'].map(lambda x: nb_dict[x])
        #feat_names.append('n_bodies')
        # number of headlines paired up with the same body text
        #nh_dict = df.groupby(['Body ID'])['headline_hash'].nunique().to_dict()
        #df['n_headlines'] = df['Body ID'].map(lambda x: nh_dict[x])
        #feat_names.append('n_headlines')
        print 'BasicCountFeatures:'
        print df
        
        # split into train, test portion and save in separate files
        train = df[~df['target'].isnull()]
        print 'train:'
        print train[['Headline_unigram','Body ID', 'count_of_Headline_unigram']]
        xBasicCountsTrain = train[feat_names].values
        outfilename_bcf_train = "train.basic.pkl"
        with open(outfilename_bcf_train, "wb") as outfile:
            cPickle.dump(feat_names, outfile, -1)
            cPickle.dump(xBasicCountsTrain, outfile, -1)
        print 'basic counting features for training saved in %s' % outfilename_bcf_train
        
        test = df[df['target'].isnull()]
        print 'test:'
        print test[['Headline_unigram','Body ID', 'count_of_Headline_unigram']]
        #return 1
        if test.shape[0] > 0:
            # test set exists
            print 'saving test set'
            xBasicCountsTest = test[feat_names].values
            outfilename_bcf_test = "test.basic.pkl"
            with open(outfilename_bcf_test, 'wb') as outfile:
                cPickle.dump(feat_names, outfile, -1)
                cPickle.dump(xBasicCountsTest, outfile, -1)
                print 'basic counting features for test saved in %s' % outfilename_bcf_test

        return 1


    def read(self, header='train'):

        filename_bcf = "%s.basic.pkl" % header
        with open(filename_bcf, "rb") as infile:
            feat_names = cPickle.load(infile)
            xBasicCounts = cPickle.load(infile)
            print 'feature names: '
            print feat_names
            print 'xBasicCounts.shape:'
            print xBasicCounts.shape
            #print type(xBasicCounts)

        return [xBasicCounts]

if __name__ == '__main__':

    cf = CountFeatureGenerator()
    cf.read()

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
