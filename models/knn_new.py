import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from models.Label_Encoder import Label_Encoder
import pickle
from sklearn import preprocessing


class KNN:
    def __init__(self, k, p):
        self.k = k#number of samples nearby
        self.p = p#distance metric
        # self.distance=distance

    def find_euclidean_distance(self,X_train, X_test):
        # the length of the datasets
        test_length = X_test.shape[0]
        train_length = X_train.shape[0]

        # Initilizing distance vector
        distance = np.zeros((test_length, train_length))

        # Computing distance between two points
        for i in range(test_length):
            for j in range(train_length):
                distance[i, j] = sum(abs(X_test[i, :] - X_train[j, :]) ** self.p)
                distance[i, j] = (distance[i, j]) ** (1 / self.p)

        return distance

    def fit(self, X_train, X_test):
        self.distance = self.find_euclidean_distance(X_train, X_test)
        # return distances

    def predict(self):
        test_length = self.distance.shape[0]  # distance length
        y_predicted = np.zeros(test_length)
        freq = []
        for i in range(test_length):
            y_indices= self.distance[i, :]
            y_indices = np.argsort(y_indices)
            k_closest_classes = y_train[y_indices[:self.k]]
            k_closest_classes =k_closest_classes.astype(int)#coverting it to int type
            frequencies = np.bincount(k_closest_classes.flatten())  # counting the frequency of the classes
            y_predicted[i] = np.argmax(frequencies)  # the most frequent class
            freq.append(frequencies / (self.k + 1))
        return y_predicted, freq
    def confusion_matrix(self,y_pred, y_test):
     #contructing empty confusion matrix with shape(7,7)
        conf_matrix = np.zeros((6, 6))

        for i, j in zip(y_test, y_pred):
            conf_matrix[i-1][j-1] = conf_matrix[i-1][j-1] + 1

        return conf_matrix.astype(int)

    def precision_calculation(self,label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def recall_calculation(self,label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def precision_macro_average(self,confusion_matrix):
        rows, columns = confusion_matrix.shape
        sum_of_precisions = 0
        for label in range(rows):
            sum_of_precisions += self.precision_calculation(label, confusion_matrix)
        return sum_of_precisions / rows

    def recall_macro_average(self,confusion_matrix):
        rows, columns = confusion_matrix.shape
        sum_of_recalls = 0
        for label in range(columns):
            sum_of_recalls += self.recall_calculation(label, confusion_matrix)
        return sum_of_recalls / columns

    def compute_confusion_matrix(self,true, pred):
        '''Computes a confusion matrix using numpy for two np.arrays
        true and pred.

        Results are identical (and similar in computation time) to:
          "from sklearn.metrics import confusion_matrix"

        However, this function avoids the dependency on sklearn.'''

        K = len(np.unique(true))  # Number of classes
        result = np.zeros((8, 8))

        for i in range(len(true)):
            result[true[i]][pred[i]] += 1

        return result.astype(int)

if __name__ == '__main__':
    #loading data
    df = pd.read_csv(r'C:\Users\sevki\PycharmProjects\485data\Data\weather_dataset1')
    df = df.sample(frac=1)
    # print(df['Severity'].unique())
    df = df.drop('Unnamed: 0', 1)
    le = preprocessing.LabelEncoder()
    df['Severity'] = df['Severity'].astype(str)
    le.fit(df['Severity'])
    df['Severity'] = le.transform(df['Severity'])

    x_train = df.iloc[:6000, :]
    x_validation = df.iloc[6000:8000, :]
    x_test = df.iloc[8000:, :]

    y_train = x_train['Labels']
    y_test = x_test['Labels']
    y_validation = x_validation['Labels']
    X_train = x_train.drop('Labels', 1)
    X_validation = x_validation.drop('Labels', 1)
    X_test = x_test.drop('Labels', 1)

    X_train = preprocessing.StandardScaler().fit_transform(X_train)
    X_test = preprocessing.StandardScaler().fit_transform(X_test)
    X_validation = preprocessing.StandardScaler().fit_transform(X_validation)

    # X_train = X_train.to_numpy()
    # X_test = X_test.to_numpy()
    # X_validation = X_validation.to_numpy()

    y_train = Label_Encoder(y_train)
    y_test = Label_Encoder(y_test)
    y_validation = Label_Encoder(y_validation)


    #Evaluating the algorithm depending on different k and p values
    k_values=[9]
    p_values=[4]
    # k_values = [7]
    # p_values = [4]
    for k in k_values:
        for p in p_values:
            print('=================================================')
            print('Results for k =',k,'and p =',p)
            knn = KNN(k=k, p=p)
            # training
            Start = time.time()
            knn.fit(X_train, X_validation)
            print('Training Time: ', time.time() - Start)
            #
            # predicting
            Start = time.time()
            y_pred, freq =knn.predict()
            # print(freq)

            with open("../scractess/test_1_"+ str(k)+ "_ "+str(p) +".txt", "wb") as fp:  # Pickling
                pickle.dump(freq, fp)

            print('Prediction Time: ', time.time() - Start)
            y_pred = y_pred.reshape(y_validation.shape[0], 1)
            #
            y_pred = y_pred.astype(int)
            y_pred = np.concatenate(y_pred)
            y_validation = y_validation.astype(int)
            print('np.unique(y_validation)',np.unique(y_validation))
            print('np.unique(y_pred)', np.unique(y_pred))
            # accuracy
            # print('Accuracy:', np.sum(y_pred == y_validation.flatten()) / y_validation.shape[0])
            conf_matr=knn.compute_confusion_matrix( y_validation.astype(int),y_pred.astype(int))
            print(conf_matr)
            # pma=knn.precision_macro_average(conf_matr)
            # print('Precision macro average:', pma)
            # rma=knn.recall_macro_average(conf_matr)
            # print('Recall macro average:', rma)


