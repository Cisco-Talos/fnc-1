'''
    super class of various feature generators
'''

class FeatureGenerator(object):

    def __init__(self, name):
        self._name = name
    
    def name(self):
        return self._name

    def process(self, data, header):
        '''
            input:
                data: pandas dataframe
            generate features and save them into a pickle file
        '''
        pass

    def read(self, header):
        '''
            read the feature matrix from a pickle file
        '''
        pass
