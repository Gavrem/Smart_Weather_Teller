import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time


class Logistic_Regression:

    def sigmoid(X, weight, b):
        z = np.array(np.dot(X, weight), dtype=np.float32) + b
        global y_predict
        y_predict =  1 / (1 + np.exp(-z))
        return y_predict

    # def square_loss(self, y_pred, target):
    #   return (np.mean(pow(y_pred - target,2)))

    def square_loss(y_pred, y_target):
        loss = (-1 * y_target* np.log(y_pred) - (1 - y_target) * np.log(1 - y_pred)).mean()
        return loss

    def log_likelihood(X, y, weight):
        z = np.dot(X, weight)
        log_like = np.sum(y * z - np.log(1 + np.exp(z)))
        return log_like

    def gradient_descent(X, y_pred, y_target):
        return np.dot(X.T, (y_pred - y_target)) / y_target.shape[0]


    def update_weight_loss( weight, lr, gradient):
        return weight - np.dot(lr, gradient.T)

    def dataset_minmax(X):
        minmax = list()
        for column in X.columns[0:]:
            col_values = X[column].values
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

    def normalize_dataset(dataset, minmax):
        i = 0
        for column in dataset.columns[0:]:
            dataset[column] = (dataset[column] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            i = i + 1

    def fit(self, X, y_train, lr):
        start_time = time.time()
        global theta, intercept
        theta = np.zeros(X.shape[1])
        intercept = 0.1
        h_prime = 0
        change = 1
        num_iter = 0
        while abs(change) > 0.00001:
            h = self.sigmoid(X, theta, intercept)
            gradient = self.gradient_descent(X, h, y_train)
            theta = self.update_weight_loss(theta, lr, gradient)
            gradient_intercept = np.mean(h - y_train)
            intercept = intercept - lr * gradient_intercept
            change = np.mean(h - h_prime)
            h_prime = h
            num_iter = num_iter +1
        print("Training time (Log Reg using Gradient descent):" + str(time.time() - start_time) + " seconds")
        print("Learning rate: {}\nIteration: {}".format(lr, num_iter))

    def predict(self,X,y_test):
        result = self.sigmoid(X, theta, intercept)
        for i in range(len(result)):
            result[i] = round(result[i], 2)
        y_test = np.float32(y_test)
        tr = 0
        fls = 0
        for i in range(len(result)):
            if abs((result[i]) - y_test[i]) >= 0.07:
                fls = fls + 1
            else:
                tr = tr + 1
        # print(tr, fls)
        # print('score', round(tr / (tr + fls),2))



        deca =[0.14, 0.29, 0.43, 0.57, 0.71, 0.86]
        for i in range(len(result)):
            if result[i] < (deca[0]+deca[1])/2 :
                result[i] = deca[0]
            for j in range(len(deca)-2):
                # print(j)
                if  (result[i] >= (deca[j]+deca[j+1])/2 ) and ( result[i] < (deca[j+1]+deca[j+2])/2):
                    result[i] = deca[j+1]
            if result[i] >= (deca[4]+deca[5])/2:
                result[i] = deca[5]

        tr = 0
        fls = 0
        for i in range(len(result)):
            if abs((result[i]) - y_test[i]) >= 0.07:
                fls = fls + 1
            else:
                tr = tr + 1
        # print(tr, fls)
        print('score', round(tr / (tr + fls),2))


        # print( (y_test == result).sum() / float(len(y_test)))
        # print(round((np.sum(result == y_test) / y_test.shape[0]),2))
        return  ((y_test == result).sum() / float(len(y_test))) , result