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
