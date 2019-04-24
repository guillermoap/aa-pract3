from src.math import gaussian

class Bayes:
    def __init__(self, data, numeric_attributes, mc, mad, mac):
        self.data = data
        self.numeric_attributes = numeric_attributes
        self.mc = mc
        self.mad = mad
        self.mac = mac

    def classify(self, element, soft_vote=False):
        probabilities = {}
        for c in self.data.clazz.unique():
            probabilities[c] = self.mc[c]
            for attribute in self.data.columns.unique():
                if attribute == 'clazz':
                    pass
                else:
                    if attribute in self.numeric_attributes:
                        x = element[attribute]
                        probabilities[c] *= gaussian(x, self.mac[c][attribute]['mean'], self.mac[c][attribute]['variance'])
                    else:
                        x = element[attribute]
                        if (x in self.mad[c][attribute]):
                            p = self.mad[c][attribute][x]
                            if p == 0:
                                p = 0.000000000001
                        else:
                            p = 0.000000000001
                        probabilities[c] *= p
        if soft_vote:
            return probabilities

        bestProb = -1
        for c, prob in probabilities.items():
            if prob > bestProb:
                bestProb = prob
                bestClass = c

        return bestClass
