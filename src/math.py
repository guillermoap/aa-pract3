import numpy as np
from scipy import stats

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
