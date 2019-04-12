import pandas
from src.tree import Tree, NumericalNode, LeafNode, CategoricalNode
from src.math import gain, threshold, numerical_gain

class Algorithm:
    def train(self, data):
        pass


class ID3(Algorithm):
    def __init__(self, data, numeric_attributes=[], specific_class=None):
        self.numeric_attributes = numeric_attributes
        self.data = data
        self.specific_class = specific_class
        self.attribute_values = dict()
        self.leaves = 0
        self.specific_class_qty = 0
        for attribute in data.columns:
            self.attribute_values[attribute] =  data[attribute].unique()

    def specific_class_probability(self):
        return (self.specific_class, self.specific_class_qty / self.leaves)

    def __set_probability_data(self, leaf):
        self.leaves += 1
        if leaf.label == leaf.specific_class:
            self.specific_class_qty += 1

    def __train(self, data):
        # cls must be the class for the entry
        if len(data.clazz.unique()) == 1: # just one class
            root = LeafNode(data.clazz.unique()[0], self.specific_class)
            if self.specific_class is not None:
                self.__set_probability_data(root)
        elif len(data.columns) == 1: # get the mode (most frequent value)
            root = LeafNode(data.iloc[0].mode().iloc[0], self.specific_class) # may be multiple most frequent values
            if self.specific_class is not None:
                self.__set_probability_data(root)
        else:
            attribute, threshold = self.best_attribute(data)
            if threshold:
                root = NumericalNode(attribute, threshold)
                examples = data.loc[data[attribute] <= threshold]
                if len(examples.index) == 0:
                    branch = LeafNode(data.clazz.mode().iloc[0], self.specific_class)
                    if self.specific_class is not None:
                        self.__set_probability_data(branch)
                else:
                    branch = self.__train(examples.drop([attribute], axis=1))
                root.add_less_equal_child(branch)
                examples = data.loc[data[attribute] > threshold]
                if len(examples.index) == 0:
                    branch = LeafNode(data.clazz.mode().iloc[0], self.specific_class)
                    if self.specific_class is not None:
                        self.__set_probability_data(branch)
                else:
                    branch = self.__train(examples.drop([attribute], axis=1))
                root.add_greater_child(branch)
            else:
                # use the mode as default for non-present attributes
                mode = data[attribute].mode().iloc[0]
                root = CategoricalNode(attribute, mode)
                for value in self.attribute_values[attribute]:
                    examples = data.loc[data[attribute] == value]
                    if len(examples.index) == 0:
                        branch = LeafNode(data.clazz.mode().iloc[0], self.specific_class)
                        if self.specific_class is not None:
                            self.__set_probability_data(branch)
                    else:
                        branch = self.__train(examples.drop([attribute], axis=1))
                    root.add_child(value, branch)
        return root

    def train(self):
        return Tree(self.__train(self.data))

    def best_attribute(self, data):
        best = None
        best_t = None
        best_gain = float('-inf')
        for attribute in data.drop(['clazz'], axis=1):
            if attribute in self.numeric_attributes:
                thresh = threshold(data, attribute, self.specific_class)
                attribute_gain = numerical_gain(data, attribute, thresh, self.specific_class)
            else:
                thresh = None
                attribute_gain = gain(data, attribute, self.specific_class)
            if attribute_gain > best_gain:
                best_gain = attribute_gain
                best = attribute
                best_t  = thresh
        return best, best_t
