from src.math import gaussian

class Bayes:
    def __init__(self, data, mc, mad, mac):
        self.data = data
        self.mc = mc
        self.mad = mad
        self.mac = mac

    def classify(self, element):
        probabilities = {}
        for c in self.data.clazz.unique():
            probabilities[c] = 1
            for attribute in self.data.columns.unique():
                if attribute == 'clazz':
                    pass
                else:
                    x = element[attribute]
                    probabilities[c] *= gaussian(x, self.mac[c][attribute]['mean'], self.mac[c][attribute]['variance'])
        
        bestProb = -1
        for c, prob in probabilities.items():
            if prob > bestProb:
                bestProb = prob
                bestClass = c

        return bestClass