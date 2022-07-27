from random import seed
from random import randrange
from csv import reader
from math import sqrt
import pandas as pd
import numpy as np
import pandas as pd
from models.Logistic_Regression import  Logistic_Regression
from Label_Encoder import Label_Encoder
from sklearn import preprocessing


df = pd.read_csv(r'C:\Users\sevki\PycharmProjects\485data\Data\weather_dataset1')
df = df.drop('Unnamed: 0', 1)
# df = df.sample(frac=1)


le = preprocessing.LabelEncoder()

df['Severity'] = df['Severity'].astype(str)
le.fit(df['Severity'])
df['Severity'] = le.transform(df['Severity'])

x_train = df.iloc[:6000,:]
x_test = df.iloc[4500:,:]

y_train = x_train['Labels']
y_test = x_test['Labels']
x_train = x_train.drop('Labels', 1)
x_test = x_test.drop('Labels', 1)

x_train = (x_train - x_train.mean()) / x_train.std(ddof=0)
x_test = (x_test - x_test.mean()) / x_test.std(ddof=0)

# converting dataframes into numpy arrays
X_train = x_train.to_numpy()
X_test = x_test.to_numpy()
y_train = Label_Encoder(y_train)
y_test = Label_Encoder(y_test)

y_train = y_train/7
y_test = y_test/7


for i  in range(len(y_train)):
    y_train[i] =round(y_train[i],2)
for i in range(len(y_test)):
    y_test[i] = round(y_test[i], 2)


X_new = X_train[0:1000, :]
y_new = y_train[0:1000]

X_test_new = X_test[0:20, :]
y_test_new = y_test[0:20]

def cross_validation_split(data,labels, n_folds):
    dataset_split = []
    dataset_split_labels=[]
    fold_size = data.shape[0] // n_folds
    for i in range(n_folds):
        fold = []
        fold_labels = []

        while len(fold) < fold_size:
            random_index = randrange(data.shape[0])
            fold.append(data[random_index])
            fold_labels.append(labels[random_index])
            data=np.delete(data, [random_index], axis=0)
            labels=np.delete(labels, [random_index], axis=0)

        dataset_split.append(fold)
        dataset_split_labels.append(fold_labels)
    return dataset_split,dataset_split_labels

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_feats = int(sqrt(len(X_new[0]) - 1))  # MODIFY
def evaluate(dataset,labels, n_folds, *args):
    folds, fold_labels = cross_validation_split(dataset,labels, n_folds)
    results = []
    print(len(folds))
    # print(folds)
    for i in range(len(folds)):
        # print('fold',fold)
        print('========================')
        X_train = []
        y_train=[]
        for j in range(len(folds)):
            if j == i:
                X_test =folds[j]
                y_test=fold_labels[j]
            else:
                X_train.append(folds[j])
                y_train.append(fold_labels[j])
        X_train = sum(X_train, [])
        y_train=sum(y_train, [])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        log_res = Logistic_Regression

        log_res.fit(log_res, X_train, y_train, 0.05)

        acc , pred = log_res.predict(log_res, X_test, y_test)
        print(acc)
        results.append(acc)
    print('Cross Validation Scores:',results)
    print('Average Cross Validation Scores:', sum(results)/n_folds)


evaluate(X_new,y_new, 5)
