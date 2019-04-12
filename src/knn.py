from src.algorithm import Algorithm
from src.math import scale
import pandas

class KnnClassificationModel():
    '''
        Classifies using K-Nearest Neighbors algorithm
        All attributes are assumed to be numerical
    '''
    def __init__(self, data, k,  normalize = True):
        self.k = k
        if normalize:
            self.data = scale(data, data.columns[:-1])
        else:
            self.data = data

    def classify(self, element):
        distances = (self.data[:, 0:-1] - element).pow(2).sum(1).pow(0.5)
        df = pandas.DataFrame({ 'distance': distances, 'clazz': self.clazz})
        sorted = df.sort_values(by=['distance'])
        return sorted.iloc[0:self.k].clazz.mode().iloc[0]

class Knn(Algorithm):
    def __init__(self, k, normalize = True):
        self.k = k
        self.normalize = normalize

    def train(self, data):
        return KnnClassificationModel(data, self.k, self.normalize)