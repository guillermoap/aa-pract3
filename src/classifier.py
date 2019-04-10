from src.algorithm import ID3

class Classifier:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.model = None

    def train(self):
        self.model = self.algorithm.train()

    def classify(self, element, vote=False):
        if vote:
            clazz, prob = self.algorithm.specific_class_probability()
            return (
                self.model.classify(element),
                { clazz: prob }
            )
        else:
            return self.model.classify(element)

