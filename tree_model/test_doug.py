import numpy as np
import pandas as pd
from collections import Counter

def perfect_score(truth_y):

    score = 0
    for i in range(truth_y.shape[0]):
        if truth_y[i] == 3: score += 0.25
        else: score += 0.75

    return score

def test():

    df = pd.read_csv('test.csv')
    targets = ['agree', 'disagree', 'discuss', 'unrelated']
    targets_dict = dict(zip(targets, range(len(targets))))
    df['target'] = map(lambda x: targets_dict[x], df['Stance'])
    y = df['target'].values
    print perfect_score(y)
    print Counter(y)

if __name__ == '__main__':
    test()
