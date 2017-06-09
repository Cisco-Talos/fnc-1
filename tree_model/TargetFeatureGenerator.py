from FeatureGenerator import *
import cPickle
import pandas as pd
from helpers import *


class TargetFeatureGenerator(FeatureGenerator):
    
    '''
        doing nothing other than returning the target variables
    '''

    def __init__(self, name='targetFeatureGenerator'):
        super(TargetFeatureGenerator, self).__init__(name)


    def process(self, df, header='train'):

        targets = df['target'].values
        outfilename_target = "%s.target.pkl" % header
        with open(outfilename_target, "wb") as outfile:
            cPickle.dump(target, outfile, -1)
        print 'targets saved in %s' % outfilename_target
        
        return targets


    def read(self, header='train'):

        filename_target = "%s.target.pkl" % header
        with open(filename_target, "rb") as infile:
            target = cPickle.load(infile)
            print 'target.shape:'
            print target.shape

        return target


