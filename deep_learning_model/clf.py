import time
from collections import defaultdict

import numpy as np
import theano
import theano.tensor as T
import theano.d3viz as d3v
import theano.gpuarray
from theano.sandbox.rng_mrg import MRG_RandomStreams as RNG

from utils import *
#data loading objects
from Vectors import *
#google vector object


def max_over_time(out):
    m = out.max(axis=-1)
    return m.reshape((m.shape[0], m.shape[1]))

class Conv1(object):
    #object wrapping for 1 dimensional convolution
    #input shape: (batch_size, n_in, 1, sequence_length)

    def __init__(self, n_in=1, n_out=1, width=1, pad=1, stride=1, dilate=1, act=T.nnet.relu):
        v = 2./float(n_in*width)
        fshape = (n_out, n_in, 1, width)
        wv = np.random.normal(0, v, fshape).astype('float32')
        bv = np.zeros(n_out).astype('float32')
        #xavier init

        w = theano.shared(wv)
        b = theano.shared(bv)
        self.w =w
        self.b =b
        self.params = [w,b]
        #setup theano vars and params

        self.act = act
        self.pad = pad
        self.stride=stride
        self.dilate=dilate
        #store anything needed for conv2d

    def __call__(self, inp):
        conv_out = conv2d(inp, self.w, subsample=(1,self.stride), border_mode=(0,self.pad), filter_dilation=(1,self.dilate))+self.b.dimshuffle('x',0,'x','x')
        #convolution + bias linear output

        act = self.act
        if act == T.nnet.relu:
            return T.nnet.relu(conv_out,.1)
        #apply non-linearity and return
        return act(conv_out)

class Conv1_Drop(object):

    def __init__(self, n_in=1, n_out=1, width=1, pad=1, stride=1, act=T.nnet.relu,rng=None,drop_prob=.5):
        v = 2./float(n_in*width)
        fshape = (n_out, n_in, 1, width)
        wv = np.random.normal(0, v, fshape).astype('float32')
        bv = np.zeros(n_out).astype('float32')

        w = theano.shared(wv)
        b = theano.shared(bv)
        self.w =w
        self.b =b
        self.params = [w,b]
        self.act = act
        self.pad = pad
        self.stride=stride
        if rng is None:
            self.rng = RNG()
        else:
            self.rng = rng
        self.drop_prob = drop_prob

        global drop_flag
        self.drop_flag = drop_flag

    def __call__(self, inp):
        conv_out = dnn_conv(inp, self.w, subsample=(1,self.stride), border_mode=(0,self.pad))+self.b.dimshuffle('x',0,'x','x')

        act = self.act
        if act == T.nnet.relu:
            nonlin = T.nnet.relu(conv_out,.1)
        nonlin = act(conv_out)

        drop_mask = self.rng.binomial(n=1, p=self.drop_prob, size=nonlin.shape)
        nonlin_drop = nonlin* T.cast(drop_mask,'float32')/self.drop_prob

        nonlin_out = T.switch(T.eq(self.drop_flag,1), nonlin_drop, nonlin)
        return nonlin_out

class Pool1(object):
    #object wrapping for 1 dimensional pooling
    #input shape: (batch_size, n_in, 1, sequence_length)
    def __init__(self, size=2):
        self.size=size

    def __call__(self, inp):
        return dnn_pool(inp, (1,self.size), (1,self.size))
        # return T.signal.pool.pool_2d(inp, (1,self.size))

def dense(inp, n_in,n_out, act=T.nnet.relu):
    #simple dense layer

    v = 2./float(n_in*n_out)
    w_v = np.random.normal(0,v,(n_in,n_out)).astype('float32')
    b_v = np.zeros(n_out).astype('float32')
    #xavier init

    w = theano.shared(w_v)
    b = theano.shared(b_v)
    lin = T.dot(inp,w)+b
    #set up theano vars, get linear output

    if act==T.nnet.relu:
        print "RELU, MANUAL LRELU"
        return T.nnet.relu(lin, .1), w,b
    elif act is None:
        return lin, w, b
    #apply non-linearity and return with params
    return act(lin), w, b

class GradClip(theano.compile.ViewOp):
  #gradient clipping between a range
    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        return [T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound) for g_out in g_outs]

def Adam(grads, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    #https://arxiv.org/abs/1412.6980
    updates = []
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


def Train(cost=None, params=None, inputs=None, outputs=None, rho=.9, givens={}, clip=False, updates={},learning_rate=.9, profile=False):
    #helper function to produce a theano training function, wrapping most of the boilerplate stuff
    if clip:
        grad_clip = GradClip(-clip,clip)
        grads = T.grad(grad_clip(cost),params)
    else:
        grads = T.grad(cost, params)
        grad_updates = Adam(grads, params)

        for u in updates:
            print u, updates[u]
            if isinstance(grad_updates, list):
                grad_updates.append((u, updates[u]))
            else:
                grad_updates[u] = updates[u]


    print 'Build train ',
    t0 = time.time()
    train_model = theano.function(inputs=inputs, outputs=outputs, updates=grad_updates, givens=givens, on_unused_input='warn', profile=profile)
    print time.time() - t0
    return train_model

def score(pred, act):
    s = 0
    if pred == act == 3:
        s+=.25
    if act != 3 and pred != 3:
        s+=.25
    if pred == act != 3:
        s+=.75
    return s

if __name__ == '__main__':
    theano.gpuarray.use('cuda1')
    v = GoogleVec()
    v.load()

    np.random.seed(2)


    n_chars = len(chars)+1
    n_classes = 4
    #static values

    n_hidden = 256
    emb_dim = 300
    #variable network hyperparameters


    body_chars = T.imatrix()
    head_chars = T.imatrix()
    target = T.ivector()
    drop_flag = T.iscalar()
    #set up symbolic inputs
    #all inputs are int



    emb_mat = v.shared()
    #get the word vector embedding matrix as a theano.shared



    head_tensor = emb_mat[head_chars].dimshuffle(0,2,'x',1)
    body_tensor = emb_mat[body_chars].dimshuffle(0,2,'x',1)
    #convert matrix of ints into tensor4 of float32s
    #character ints are indices
    #output shape (batch_size, emb_dim, 1, sequence_length)

    c0 = Conv1_Drop(n_in=emb_dim, n_out=n_hidden, width=3,pad=3)
    c1 = Conv1_Drop(n_in=n_hidden, n_out=n_hidden, width=3,pad=3)
    c2 = Conv1_Drop(n_in=n_hidden, n_out=n_hidden*2, width=3,pad=3)
    c3 = Conv1_Drop(n_in=n_hidden*2, n_out=n_hidden*2, width=3,pad=1)
    c4 = Conv1_Drop(n_in=n_hidden*2, n_out=n_hidden*3, width=3,pad=1)
    #1d conv layers, all with width=3, hidden size increases with depth
    p0 = Pool1()
    p1 = Pool1()
    p2 = Pool1()
    #a couple pooling layers

    c0_o = c0(body_tensor)
    p0_o = p0(c0_o)
    c1_o = c1(p0_o)
    p1_o = p1(c1_o)
    c2_o = c2(p1_o)
    p2_o = p2(c2_o)
    c3_o = c3(p2_o)
    body_conv = c4(c3_o)
    #run the bodies through all conv ops

    c0_o = c0(head_tensor)
    p0_o = p0(c0_o)
    c1_o = c1(p0_o)
    p1_o = p1(c1_o)
    c2_o = c2(p1_o)
    p2_o = p2(c2_o)
    c3_o = c3(p2_o)
    head_conv = c4(c3_o)
    #run the heads through all conv ops


    head_vec = max_over_time(head_conv)
    body_vec = max_over_time(body_conv)
    #head/body_conv still variable length, take max feature value to get fixed-length vector


    feature_vec = T.concatenate([head_vec, body_vec],axis=1)
    #join head/body features

    d0, w0, b0 = dense(feature_vec, n_in=n_hidden*3*2, n_out=n_hidden*4)
    d1, w1, b1 = dense(d0, n_in=n_hidden*4, n_out=n_hidden*4)
    d2, w2, b2 = dense(d1, n_in=n_hidden*4, n_out=n_hidden*4)
    s0, ws, bs = dense(d2, n_in=n_hidden*4, n_out=n_classes, act=T.nnet.softmax)
    #create fully-connected relu net
    #output is softmax since problem is multi-class but not multi-label

    pred = T.argmax(s0,axis=1)
    err = T.neq(pred,target.flatten()).mean()
    #predictions/error rate

    cost = T.nnet.categorical_crossentropy(s0, target).mean()
    #correct cross_entropy for softmax

    params = [w0,b0,w1,b1,w2,b2,ws,bs]+c0.params+c1.params+c2.params+c3.params+c4.params
    #collect all parameters to learn via gradient descent



    news = News(stances='train.csv', bodies='train_bodies.csv', vecs=v)
    #FNC Baseline training set
    val_news = News(stances='test.csv', bodies='train_bodies.csv', vecs=v)
    #FNC baseline validation set
    test_news = News(stances='test_stances_unlabeled.csv', vecs=v, bodies='test_bodies.csv')
    #FNC final test set


    test = theano.function([head_chars, body_chars, target], pred, on_unused_input='ignore', givens={drop_flag:np.int32(0)})
    probs = theano.function([head_chars, body_chars, target], s0, on_unused_input='ignore', givens={drop_flag:np.int32(0)})
    #compile function that just returns predictions
    train = Train(cost=cost, params=params, inputs=[head_chars, body_chars, target], outputs=cost, profile=False, givens={drop_flag:np.int32(1)})
    #compile function that returns cost term and performs 1 step of gradient descent


    costs = []
    right = defaultdict(int)
    wrong = defaultdict(int)
    t0 = time.time()
    n = 32
    eval_test = True
    output_fn = 'deepoutput.csv'

    if eval_test:
        with open('params_final.p','r') as f:
            params1=pickle.load(f)
        for o,n in zip(params, params1):
            assert o.get_value().shape == n.get_value().shape
            o.set_value(n.get_value())
        #extremely simple model persistance, pickle our list of params
        #if we attempt to load weights that don't match the shape, make some noise

        with open(output_fn,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Headline','BodyID','Agree', 'Disagree','Discuss','Unrelated'])
            #set up our csv
            i = 0
            for head,body,stance in test_news.validate():
                #iterate through the test_news set and output probabilities to csv
                t = []
                h,b,s = test_news.headlines[i]
                t.append(h)
                t.append(b)
                sp = probs(head,body,stance)
                #tmp.append(sp[0])
                t.extend(sp[0])
                writer.writerow(t)

                i+=1
                if i%50==0:
                    print i
        exit()

    best_score = 0
    #simple training loop, draw n samples and perform 1 GD update
    #every so often print stats/save params
    for i in xrange(35000000):

        head,body, stance = news.sample(n=n)
        try:
            #Catch OOM/small size problems and just continue
            c = train(head, body, stance)
            costs.append(c)
        except:
            pass


        if i%500==0:
            print i, np.mean(costs), time.time() - t0
            t0 = time.time()
            costs = []

        if i%5000==0 and i>10:
            #every 5000 steps, check against the validation set
            #if we have a new high score, save the params
            m = 0
            b = 0
            print i, time.time() - t0
            for head,body,stance in val_news.validate():
                preds = test(head, body, stance)
                for p, a in zip(preds, stance):

                    m+= score(p, a)
                    b+= score(a, a)

                    if p == a:
                        right[p]+=1
                    else:
                        wrong[p]+=1

            print m, b
            print ' '
            t0 = time.time()
            if m>best_score:
                print 'SAVED ---------------------------',m
                best_score = m
                with open('params.p','wb') as f:
                   pickle.dump(params, f)

            right = defaultdict(int)
            wrong = defaultdict(int)

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
