import pandas as pd
from models.Logistic_Regression import  Logistic_Regression
from Label_Encoder import Label_Encoder
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix( y_pred, y_test):
    # constructing empty confusion matrix
    conf_matrix = np.zeros((7, 7))

    for i, j in zip(y_test, y_pred):
        conf_matrix[i-1][j-1] = conf_matrix[i-1][j-1] + 1

    return conf_matrix.astype(int)

df = pd.read_csv(r'C:\Users\sevki\PycharmProjects\485data\Data\weather_dataset1')
df = df.drop('Unnamed: 0', 1)
# df = df.sample(frac=1)
# print(df['Severity'].unique())

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
x_train = x_train.drop('Labels', 1)
x_validation = x_validation.drop('Labels', 1)
x_test = x_test.drop('Labels', 1)

pd.set_option('display.max_columns', None)

label = y_train.copy()
label2 = y_test.copy()


y_train = Label_Encoder(y_train)
y_test = Label_Encoder(y_test)
y_validation = Label_Encoder(y_validation)


x_train = (x_train - x_train.mean()) / x_train.std(ddof=0)
x_test = (x_test - x_test.mean()) / x_test.std(ddof=0)
x_validation = (x_validation - x_validation.mean()) / x_validation.std(ddof=0)

y_test2= y_test
y_train = y_train/7
y_test = y_test/7
y_validation2 = y_validation
y_validation = y_validation/7



for i  in range(len(y_train)):
    y_train[i] =round(y_train[i],2)
for i in range(len(y_test)):
    y_test[i] = round(y_test[i], 2)
for i in range(len(y_validation)):
    y_validation[i] = round(y_validation[i], 2)

log_res = Logistic_Regression


lr_rates = [0.065   ] #list(np.array(list(range(0,100,1)))/1000)
points = []
for lr_rate in lr_rates:

    log_res.fit(log_res,x_train,y_train,lr_rate)
    acc ,y_pred = log_res.predict(log_res,x_test,y_test)
    acc = acc * 100
    points.append([acc, lr_rate])

df1=pd.DataFrame(points,columns=["x","y"])
cl1 = plt.scatter(df1.y,df1.x)
plt.xlabel('learning rate')
plt.ylabel('accuray (%)')
# plt.show()

# y_pred = np.around(y_pred*7)
# y_pred = y_pred.astype(int)
#
# print("Accuracy =", acc)
# print("Confusion Matrix for Logistic \n regression at learning rate  = " + str(lr_rates[0]))
# print((confusion_matrix(y_pred,y_test2)))

# p1 = PCA
# pcs = p1.fit(x_train,label)
# pcs = pcs.drop('label',1)
# log_res.fit(log_res,pcs,y_train,1000,0.1)
#
# pcs2 = p1.fit(x_test,label2)
# pcs2 = pcs2.drop('label',1)
# log_res.predict(log_res,pcs2,y_test)