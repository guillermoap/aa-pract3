import warnings
import pandas
import sys

from src.knn import Knn
from src.bayes import bayes
from src.classifier import Classifier, ID3
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from optparse import OptionParser
from src.optimize import optimize

warnings.filterwarnings('ignore')
IRIS_NUMERIC_ATTRIBUTES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
COV_TYPE_NUMERIC_ATTRIBUTES = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points', 'Hillshade_Mean']

ID3_ALGORITHM = 'id3'
KNN_1_ALGORITHM = '1nn'
KNN_3_ALGORITHM = '3nn'
KNN_7_ALGORITHM = '7nn'
NAIVE_BAYES_ALGORITHM = 'nbayes'

def create_classifier(train, algorithm, numeric_attributes=[], specific_class=None):
    if algorithm == ID3_ALGORITHM:
        print('algoritmo id3')
        return Classifier(ID3(train, numeric_attributes, specific_class))
    elif algorithm == KNN_1_ALGORITHM:
        print('algoritmo knn. k = 1')
        return Classifier(Knn(1, train, specific_class))
    elif algorithm == KNN_3_ALGORITHM:
        print('algoritmo knn. k = 3')
        return Classifier(Knn(3, train, specific_class))
    elif algorithm == KNN_7_ALGORITHM:
        print('algoritmo knn. k = 7')
        return Classifier(Knn(7, train, specific_class))
    elif algorithm == NAIVE_BAYES_ALGORITHM:
        print('algoritmo naive-bayes')
        return Classifier(bayes(train, numeric_attributes, specific_class))
    else:
        return None

def parte_a(train, test, numeric_attributes=IRIS_NUMERIC_ATTRIBUTES, algorithm = ID3_ALGORITHM):
    classifier = create_classifier(train, algorithm, numeric_attributes)
    classifier.train()

    actual = []
    predicted = []
    for _, elem in test.iterrows():
        actual.append(elem.clazz)
        predicted.append(classifier.classify(elem.iloc[0:-1]))
    output_results(title='PARTE A', actual=actual, predicted=predicted)

def parte_b(train, test, numeric_attributes=IRIS_NUMERIC_ATTRIBUTES, algorithm = ID3_ALGORITHM):
    classifiers = []
    classes = train.clazz.unique()
    idx = 1
    rest = train
    length = len(classes)
    for clazz in classes:
        if algorithm == NAIVE_BAYES_ALGORITHM:
            amount = 1 / length
            length -= 1
            if idx != len(classes):
                rest, train = split_data(rest, amount)
            else:
                train = rest
        classifier = create_classifier(train, algorithm, numeric_attributes, clazz)
        classifier.train()
        classifiers.append(classifier)
        idx += 1

    actual = []
    predicted = []
    for _, elem in test.iterrows():
        actual.append(elem.clazz)
        if algorithm == NAIVE_BAYES_ALGORITHM:
            predicted.append(soft_vote_classify(classifiers, elem[0:-1], classes))
        else:
            predicted.append(vote_classify(classifiers, elem[0:-1]))
    output_results(title='PARTE B', actual=actual, predicted=predicted)

def parte_c(train, test, numeric_attributes=COV_TYPE_NUMERIC_ATTRIBUTES, algorithm = ID3_ALGORITHM):
    parte_a(train, test, numeric_attributes, algorithm)
    parte_b(train, test, numeric_attributes, algorithm)

def output_results(title, actual, predicted):
    print(f'######### {title} #########')
    print('score: micro, macro')
    print(f"precision: {precision_score(actual, predicted, average='micro')}, {precision_score(actual, predicted, average='macro')}")
    print(f"recall: {recall_score(actual, predicted, average='micro')}, {recall_score(actual, predicted, average='macro')}")
    print(f"f1: {f1_score(actual, predicted, average='micro')}, {f1_score(actual, predicted, average='macro')}")

def vote_classify(clasiffiers, element):
    results = []
    for classifier in clasiffiers:
        classification, classification_prob = classifier.classify(element, vote=True)
        results.append((classification, classification_prob))

    classifications = list(map(lambda elem: elem[1], filter(lambda elem: elem[0], results))) # classifications = [classification_prob] where classification is True

    if classifications:
        result = min(classifications, key=(lambda elem: list(elem.values())[0]))
        # we keep the class with least probability of ocurrance
    else:
        classifications = list(map(lambda elem: elem[1], results))
        result = max(classifications, key=(lambda elem: list(elem.values())[0]))
        # we keep the class with least probability of not ocurrance
    result = list(result)[0] # keep the key which is the class name
    return result

def soft_vote_classify(classifiers, element, classes):
    results = {}
    length = len(classifiers)
    for clazz in classes:
        results[clazz] = 0
        for classifier in classifiers:
            probabilities = classifier.soft_vote_classify(element)
            results[clazz] += probabilities[clazz]
        results[clazz] /= length

    return max(results, key=results.get)

def split_data(data, amount):
    train, test = train_test_split(data, test_size=amount)
    return train, test

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-p', '--covtype-percent', dest='covtype_percent',
     help='porcentaje, en el intervalo [0,1], a utilizar del cover type dataset')
    parser.add_option('-a', '--algorithm', dest='algorithm',
     help="Algoritmo a utilizar. Puede ser 'id3', '1nn', '3nn', '7nn' o 'nbayes' sin comillas")
    options, args = parser.parse_args()
    if options.covtype_percent:
        training_size_covtype = float(options.covtype_percent)
        print(f'Usando {training_size_covtype * 100} por ciento del cover type dataset')
    else:
        print('Usando 100 por ciento del cover type dataset')
        print('Esto puede demorar. Puede pasar como argumento la proporcion del dataset covtype que desea usar')
        training_size_covtype = 1
    if not options.algorithm:
        print('No se selecciono algoritmo. Se usara id3')
        algorithm = ID3_ALGORITHM
    else:
        algorithm = options.algorithm
    data = pandas.read_csv('./iris.csv')
    train, test = train_test_split(data, test_size=0.2)
    print("### IRIS ###")
    parte_a(train=train, test=test, algorithm=algorithm)
    parte_b(train=train, test=test, algorithm=algorithm)
    optimize()

    print("### COVER_TYPE LOG (opt.log.csv) ###")
    data = pandas.read_csv('./covtype.data.opt.log.csv')
    if training_size_covtype < 1:
        pseudo_train, dataset = train_test_split(data, test_size=training_size_covtype)
    else:
        dataset = data
    train, test = train_test_split(dataset, test_size=0.2)
    parte_c(train, test, algorithm = algorithm)
