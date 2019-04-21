from src.algorithm import Algorithm
from src.math import scale
from scipy.spatial import cKDTree
import pandas

class KnnSortingClassificationModel():
    '''
        Classifies using K-Nearest Neighbors algorithm
        Sorts the dataset to find the k nearest neighbours in 
        O(|data| * log |data|)
        All attributes are assumed to be numerical
    '''
    def __init__(self, data, k, specific_class=None):
        self.k = k
        if specific_class:
            self.data = data.iloc[:, 0:-1]
            self.data['clazz'] = data.clazz.apply(lambda x: x == specific_class)
        else:
            self.data = data
    def classify(self, element):
        distances = (self.data.iloc[:, 0:-1] - element).pow(2).sum(1).pow(0.5)
        df = pandas.DataFrame({ 'distance': distances, 'clazz': self.data.clazz })
        sorted = df.sort_values(by=['distance'])
        return sorted.iloc[0:self.k].clazz.mode().iloc[0]

class KnnKdTreeClassificationModel():
    '''
        Classifies using K-Nearest Neighbors algorithm.

        Uses a kdtree to find the k nearest neighbours in O(k * log |data|).
        Remembering k is actually a constant, this means is O(log |data|).

        All attributes are assumed to be numerical.
    '''
    def __init__(self, data, k, specific_class=None):
        self.k = k
        if specific_class:
            # This columns are areference to original dataframe,
            # If we do not modify them, we wont mutate it
            self.data = data.iloc[:, 0:-1]
            # We add a column, but do not modify an existing one
            self.data['clazz'] = data.clazz.apply(lambda x: x == specific_class)
        else:
            self.data = data
            # cKDTree is a native (and so more efficient) version of KDtree
        self.kdtree = cKDTree(self.data.iloc[:, 0:-1])

    def classify(self, element):
        distances, indexes = self.kdtree.query([element], self.k)
        if self.k > 1:
            indexes = indexes[0]
        knn = self.data.iloc[indexes]
        return knn.clazz.mode().iloc[0]

class Knn(Algorithm):
    def __init__(self, k, data, specific_class=None, kdtree = True):
        self.k = k
        self.data = data
        self.specific_class = specific_class
        self.kdtree = kdtree

    def specific_class_probability(self):
        return self.specific_class, 1/len(self.data.clazz.unique()) # Puede y debe rendir más

    def train(self):
        if self.kdtree:
            return KnnKdTreeClassificationModel(self.data, self.k, self.specific_class)
        else:
            return KnnSortingClassificationModel(self.data, self.k, self.specific_class)