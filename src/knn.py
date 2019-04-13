from src.algorithm import Algorithm
from src.math import scale
import pandas

class KnnClassificationModel():
    '''
        Classifies using K-Nearest Neighbors algorithm
        All attributes are assumed to be numerical
    '''
    def __init__(self, data, k):
        self.k = k
        self.data = data

    def classify(self, element):
        distances = (self.data.iloc[:, 0:-1] - element).pow(2).sum(1).pow(0.5)
        
        df = pandas.DataFrame({ 'distance': distances, 'clazz': self.data.clazz})
        sorted = df.sort_values(by=['distance'])
        return sorted.iloc[0:self.k].clazz.mode().iloc[0]

class Knn(Algorithm):
    def __init__(self, k, data):
        self.k = k
        self.data = data

    def train(self):
        return KnnClassificationModel(self.data, self.k)