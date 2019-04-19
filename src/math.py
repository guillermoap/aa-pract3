import numpy as np
from scipy import stats
import pandas
import math

def entropy(classes, specific_class=None):
    '''
        PRECONDITION: data are just the classes
        Based on https://gist.github.com/jaradc/eeddf20932c0347928d0da5a09298147
    '''
    if specific_class:
        _, counts = np.unique(classes.map(lambda x: x == specific_class), return_counts=True)
    else:
        _, counts = np.unique(classes, return_counts=True)
    return stats.entropy(counts, base=2)


def gain(data, attribute, specific_class=None):
    gain = entropy(data.clazz, specific_class)
    total_size = len(data.index)
    for value in data[attribute].unique():
        value_data = data.loc[data[attribute] == value]
        value_size = len(value_data.index)
        value_entropy = entropy(value_data.clazz, specific_class)
        gain -= (value_size/total_size) * value_entropy
    return gain

def numerical_gain(data, attribute, value, specific_class=None):
    gain = entropy(data.clazz, specific_class)
    total_size = len(data.index)

    leq_data = data.loc[data[attribute] <= value]
    leq_size = len(leq_data.index)
    leq_entropy = entropy(leq_data.clazz, specific_class)
    gain -= (leq_size/total_size) * leq_entropy


    g_data = data.loc[data[attribute] > value]
    g_size = len(g_data.index)
    g_entropy = entropy(g_data.clazz, specific_class)

    gain -= (g_size/total_size) * g_entropy

    return gain

def optimized_num_gain(data, cut, specific_class=None):
    gain = entropy(data.clazz, specific_class)
    total_size = len(data.index)

    leq_data = data.iloc[:cut]
    leq_size = len(leq_data.index)
    leq_entropy = entropy(leq_data.clazz, specific_class)
    gain -= (leq_size/total_size) * leq_entropy


    g_data = data.iloc[cut:]
    g_size = len(g_data.index)
    g_entropy = entropy(g_data.clazz, specific_class)

    gain -= (g_size/total_size) * g_entropy

    return gain

def threshold(data, attribute, specific_class=None):
    sorted_data = data.sort_values(by = attribute)
    i_uniques = sorted_data[attribute].index.unique().sort_values()
    cut = 0
    max_gain = optimized_num_gain(data, cut, specific_class=None)
    for i in i_uniques[:-1]:
        current_gain = optimized_num_gain(sorted_data, i, specific_class=None)
        if current_gain > max_gain:
            max_gain = current_gain
            cut = i
    return sorted_data[attribute].iloc[cut]


def scale(data, attributes = []):
    '''
        Scale usin min and max
        Useful for data normalization
    '''    
    for attribute in attributes:
        min_attribute = data[attribute].min()
        max_attribute = data[attribute].max()
        max_min_diff = max_attribute - min_attribute
        data[attribute] = (data[attribute] - min_attribute) / max_min_diff

def mean(data, specific_class, attribute):
    return (data.loc[data['clazz']==specific_class][attribute].mean())

def variance(data, specific_class, attribute):
    return (data.loc[data['clazz']==specific_class][attribute].var())

def gaussian(x, mean, variance):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*variance)))
    return (1 / (math.sqrt(2*math.pi*variance))*exponent)