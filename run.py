import warnings
from src.knn import Knn
from src.classifier import Classifier, ID3
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas
import sys, getopt
from src.optimize import optimize

warnings.filterwarnings('ignore')
IRIS_NUMERIC_ATTRIBUTES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
COV_TYPE_NUMERIC_ATTRIBUTES = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points', 'Hillshade_Mean']

def parte_a(train, test, numeric_attributes=IRIS_NUMERIC_ATTRIBUTES):
    classifier = Classifier(ID3(train, numeric_attributes))
    classifier.train()
    actual = []
    predicted = []
    for _, elem in test.iterrows():
        actual.append(elem.clazz)
        predicted.append(classifier.classify(elem.drop(columns=['clazz'], axis=1)))
    output_results(title='PARTE A', actual=actual, predicted=predicted)

def parte_b(train, test, numeric_attributes=IRIS_NUMERIC_ATTRIBUTES):
    classifiers = []
    classes = train.clazz.unique()
    idx = 1
    for clazz in classes:
        classifier = Classifier(ID3(train, numeric_attributes, specific_class=clazz))
        classifier.train()
        classifiers.append(classifier)
        idx += 1

    actual = []
    predicted = []
    for _, elem in test.iterrows():
        actual.append(elem.clazz)
        predicted.append(vote_classify(classifiers, elem.drop(columns=['clazz'], axis=1)))
    output_results(title='PARTE B', actual=actual, predicted=predicted)

def parte_c(train, test, numeric_attributes=COV_TYPE_NUMERIC_ATTRIBUTES):
    parte_a(train, test, numeric_attributes)
    parte_b(train, test, numeric_attributes)

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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        training_size_covtype = float(sys.argv[1]) 
        print(f'Usando {training_size_covtype * 100} por ciento del cover type dataset')    
    else:
        print('Usando 100 por ciento del cover type dataset')
        print('Esto puede demorar. Puede pasar como argumento la proporcion del dataset covtype que desea usar')
        print('Esto es, por ejemplo:')
        print('python3 run.py 0.02')
        training_size_covtype = 0.02    
    data = pandas.read_csv('./iris.csv')
    train, test = train_test_split(data, test_size=0.2)
    print("### IRIS ###")
    parte_a(train=train, test=test)
    parte_b(train=train, test=test)
    optimize()
    print("### COVER_TYPE NUMERIC (opt.csv) ###")
    data = pandas.read_csv('./covtype.data.opt.csv')
    pseudo_train, dataset = train_test_split(data, test_size=training_size_covtype)
    train, test = train_test_split(dataset, test_size=0.2)
    parte_c(train, test)

    print("### COVER_TYPE LOG (opt.log.csv) ###")
    data = pandas.read_csv('./covtype.data.opt.log.csv')
    pseudo_train, dataset = train_test_split(data, test_size=training_size_covtype)
    train, test = train_test_split(dataset, test_size=0.2)
    parte_c(train, test)
