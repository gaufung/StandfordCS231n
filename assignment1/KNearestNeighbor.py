'''
Implement of the k nerarest neighbor classifier
'''
from collections import Counter
import numpy as np
class KNearestNeighbor(object):
    '''
    The k nearest neightbor
    '''
    def __init__(self, k=2, distancer=None):
        '''
        constructor
        '''
        self.k = k
        self.distance = distancer
        if self.distance == None:
            self.distance = L1Distance()
    def fit(self, X, y):
        '''
        read the train data
        X: the input featues with shape: N * D, where N is the No. of train samples and D is the No. of the features
        y: the input labels with shape: N * 1, where N is the No. of train smaples
        '''
        self.Xtr = X
        self.ytr = y
    def predict(self, X):
        '''
        predict of the predict
        '''
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        #print(X.shape)
        for i in range(num_test):
            x = X[i, :]
            distances = np.sum(self.distance.run(self.Xtr - x), axis=1)
            min_indexes = np.argsort(distances)[0:self.k]
            y_candidate_labels = [self.ytr[index] for index in min_indexes]
            counter = Counter(y_candidate_labels)
            Ypred[i] = counter.most_common(1)[0][0]
        return Ypred

class distancer(object):
    def run(self,x):
        pass

class L1Distance(distancer):
    def run(self,x):
        return np.abs(x)

class L2Distance(distancer):
    def run(self, x):
        return np.square(x)

