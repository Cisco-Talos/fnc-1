import csv

import cPickle as pickle
import gzip
from time import time

import numpy as np
import theano
import theano.tensor as T

try:
	from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
except:
	from theano.gpuarray.dnn import dnn_conv, dnn_pool


from theano.tensor.nnet import conv2d


from collections import defaultdict


from Vectors import *



chars = set([chr(i) for i in range(32,128)])
#character whitelist
stances = {'agree':0,'disagree':1,'discuss':2,'unrelated':3}
#set up some values for later


def transform(text):
    #convert a string into a np array of approved character indices, starting at 0
    return np.array([ord(i)-32 for i in text if i in chars])

def pad_char(text, padc=-1):
    #take a set of variable length arrays and convert to a matrix with specified fill value
    maxlen = max([len(i) for i in text])
    tmp = np.ones((len(text), maxlen),dtype='int32')
    tmp.fill(padc)
    for i in range(len(text)):
        data = text[i]
        tmp[i,:len(data)]=data
    return tmp



def split():
    #split data into train/test, not used
    train = []
    test = []
    with open('train_ids.txt','r') as f:
        train_sets = set([int(i.strip()) for i in f.readlines()])
    with open('test_ids.txt','r') as f:
        test_sets = set([int(i.strip()) for i in f.readlines()])
    print len(train_sets), len(test_sets)
    with open('train_stances.csv','r') as f:
        reader = csv.reader(f)
        reader.next()
        for l in reader:

            if int(l[1]) in train_sets:

                train.append(l)
            else:
                test.append(l)

    print len(train), len(test)

    for dat, fn in zip([train, test],['train.csv','test.csv']):
        with open(fn,'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['header'])
            for l in dat:
                writer.writerow(l)

def proc_bodies(fn):
    #process the bodies csv into arrays
    tmp = {}
    with open(fn,'r') as f:
        reader = csv.reader(f)
        reader.next()
        for line in reader:
            bid, text = line
            tmp[bid]=text
    return tmp

class News(object):
    #object for processing and presenting news to clf

    def __init__(self, stances='train_stances.csv',bodies='train_bodies.csv',vecs=None):
        #process files into arrays, etc
        self.bodies = proc_bodies(bodies)
        self.headlines = []
        self.vecs=vecs

        with open(stances,'r') as f:
            reader = csv.reader(f)
            reader.next()
            for line in reader:
                if len(line)==2:
                    hl, bid = line
                    stance = 'unrelated'
                else:
                    hl, bid, stance = line
                self.headlines.append((hl,bid,stance))


        self.n_headlines = len(self.headlines)


    def get_one(self, ridx=None):
        #select a single sample either randomly or by index
        if ridx is None:
            ridx = np.random.randint(0,self.n_headlines)
        head = self.headlines[ridx]
        body = self.bodies[head[1]]


        return head, body


    def sample(self, n=16, ridx=None):
        #select a batch of samples either randomly or by index
        heads = []
        bodies = []
        stances_d = []
        if ridx is not None:
            for r in ridx:
                head, body_text = self.get_one(r)
                head_text, _, stance = head
                heads.append(head_text)
                bodies.append(body_text)
                stances_d.append(stances[stance])
        else:
            for i in range(n):
                head, body_text = self.get_one()
                head_text, _, stance = head
                heads.append(head_text)
                bodies.append(body_text)
                stances_d.append(stances[stance])


        heads = self.vecs.transform(heads)
        bodies = self.vecs.transform(bodies)
        stances_d = np.asarray(stances_d, dtype='int32')
        #clean up everything and return it

        return heads, bodies, stances_d


    def validate(self):
        #iterate over the dataset in order
        for i in xrange(len(self.headlines)):
            yield self.sample(ridx=[i])



if __name__ == '__main__':
    v = GoogleVec()

    val_news = News(stances='test_stances_unlabeled.csv', vecs=v, bodies='test_bodies.csv')
    v.load()

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
