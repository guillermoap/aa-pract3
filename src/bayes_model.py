from src.math import gaussian

class Bayes:
    def __init__(self, data, mc, mad, mac):
        self.data = data
        self.mc = mc
        self.mad = mad
        self.mac = mac

    def classify(self, element):
        print ('classifying..')
        pass
