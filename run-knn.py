from src.knn import Knn
from src.classifier import Classifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from src.math import scale

import pandas


def output_results(title, actual, predicted):
    print(f'######### {title} #########')
    print('score: micro, macro')
    print(f"precision: {precision_score(actual, predicted, average='micro')}, {precision_score(actual, predicted, average='macro')}")
    print(f"recall: {recall_score(actual, predicted, average='micro')}, {recall_score(actual, predicted, average='macro')}")
    print(f"f1: {f1_score(actual, predicted, average='micro')}, {f1_score(actual, predicted, average='macro')}")


data = pandas.read_csv('./iris.csv')
scale(data, data.columns[:-1])
train, test = train_test_split(data, test_size=0.2)

cls1 = Classifier(Knn(1, train))
cls3 = Classifier(Knn(3, train))
cls7 = Classifier(Knn(7, train))

cls1.train()
cls3.train()
cls7.train()

actual = []
predicted = []
for _, elem in test.iterrows():
    actual.append(elem.clazz)
    predicted.append(cls1.classify(elem.drop(columns=['clazz'], axis=1)))
output_results(title='PARTE A', actual=actual, predicted=predicted)

actual = []
predicted = []
for _, elem in test.iterrows():
    actual.append(elem.clazz)
    predicted.append(cls3.classify(elem.drop(columns=['clazz'], axis=1)))
output_results(title='PARTE A', actual=actual, predicted=predicted)

actual = []
predicted = []
for _, elem in test.iterrows():
    actual.append(elem.clazz)
    predicted.append(cls7.classify(elem.drop(columns=['clazz'], axis=1)))
output_results(title='PARTE A', actual=actual, predicted=predicted)
