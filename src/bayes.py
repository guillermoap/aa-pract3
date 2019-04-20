from src.algorithm import Algorithm
from src.math import mean, variance
from src.bayes_model import Bayes
import pandas

class bayes(Algorithm):
    def __init__(self, data, numeric_attributes=[], specific_class=None):
        self.data = data
        self.numeric_attributes = numeric_attributes
        self.attribute_values = dict()
        self.specific_class = specific_class

        for attribute in data.columns:
            self.attribute_values[attribute] =  data[attribute]
        # print (self.attribute_values)


    def qByClass(self):
        q = {}
        for c in (self.data.clazz):
            if c in q:
                q[c] = q[c]+1
            else:
                q[c] = 1
        return q

    def separateByClass(self):
        separated = {} # dict with classes and vectors in each class
        for row in (self.data.itertuples(index=False)):
            if row[-1] not in separated:
                separated[row[-1]] = []
            separated[row[-1]].append(row)
        # print (separated)
        return separated

    def train(self):
        mc = self.qByClass() # a priori, por ahora solo cantidades, al final se normaliza
        mad = {} # matriz atributos discretos
        mac = {} # matriz atributos continuos (con mean y variance)
        for c in self.data.clazz.unique():
            mac[c] = {}
            mad[c] = {}
            for attribute in self.data.columns.unique():
                if attribute == 'clazz':
                    pass
                else:
                    # print (attribute)
                    if attribute in self.numeric_attributes:
                        mac[c][attribute] = {}
                        # atributo continuo
                        # calcular media
                        m = mean(self.data, c, attribute)
                        # calcular varianza
                        v = variance(self.data, c, attribute)
                        mac[c][attribute]['mean'] = m
                        mac[c][attribute]['variance'] = v
                    else:
                        # atributo discreto
                        # print ('discreto')
                        mad[c][attribute] = {}
                        for value in self.attribute_values[attribute].unique():
                            mad[c][attribute][value] = len(self.data[self.data['clazz']==c][self.data[attribute]==value])/mc[c]
                            # print (value)

        # normalize
        for key, val in mc.items():
            mc[key] = (val/len(self.data))
        # print('mc')
        # print (mc)
        # print('mac')
        # print (mac)
        # print('mad')
        # print (mad)

        return Bayes(self.data, self.numeric_attributes, mc, mad, mac)

    def specific_class_probability(self):
        pass

