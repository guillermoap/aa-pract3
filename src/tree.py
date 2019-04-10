class Node:
    def __init__(self, label):
        self.label = label

    def classify(self, element):
        pass


class LeafNode(Node):
    def __init__(self, classification, specific_class):
        super().__init__(classification)
        self.specific_class = specific_class

    def classify(self, element=None):
        if self.specific_class is not None:
            return self.specific_class == self.label
        else:
            return self.label

    def to_dict(self):
        return self.label

    def __str__(self):
        return str(self.to_dict())

class NumericalNode(Node):
    def __init__(self, attribute, value):
        super().__init__(attribute)
        self.greater = None
        self.less_equal = None
        self.value = value

    def add_greater_child(self, node):
        self.greater = node

    def add_less_equal_child(self, node):
        self.less_equal = node

    def classify(self, element):
        if element[self.label] <= self.value:
            return self.less_equal.classify(element)
        else:
            return self.greater.classify(element)
    def to_dict(self):
        return None

    def __str__(self):
        return str(self.to_dict())

class CategoricalNode(Node):
    def __init__(self, attribute, mode):
        super().__init__(attribute)
        self.children = dict()
        self.mode = mode

    def add_child(self, value, node):
        self.children[value] = node

    def classify(self, element):
        value = element.loc[self.label]
        if value in self.children:
            return self.children[value].classify(element)
        else:
            return self.children[self.mode].classify(element)

    def to_dict(self):
        return None

    def __str__(self):
        return str(self.to_dict())

class Tree:
    def __init__(self, root):
        self.root = root

    def classify(self, element):
        return self.root.classify(element)

    def to_dict(self):
        return self.root.to_dict()

    def __str__(self):
        return str(self.to_dict())
