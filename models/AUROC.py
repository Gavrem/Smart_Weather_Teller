import pandas as pd
from sklearn import preprocessing
from models.Label_Encoder import Label_Encoder
import numpy as np
import matplotlib.pyplot as plt
import pickle



df = pd.read_csv(r'C:\Users\sevki\PycharmProjects\485data\Data\weather_dataset1')

# print(df['Severity'].unique())
df = df.drop('Unnamed: 0', 1)

le = preprocessing.LabelEncoder()
df['Severity'] = df['Severity'].astype(str)
le.fit(df['Severity'])
df['Severity'] = le.transform(df['Severity'])

x_validation = df.iloc[6000:8000, :]

y_validation = x_validation['Labels']
y_validation = Label_Encoder(y_validation)

with open("../scractess/test_1_11_ 2.txt", "rb") as fp:   # Unpickling
    predict = pickle.load(fp)


for i in range(len(predict)):
    if len(predict[i])<7:
        point = len(predict[i])
        for j in range(7-point):
            # predict[i][j+point] = 0
            predict[i]= np.append(predict[i], np.array([0]))


# for i in predict:
#     print(i)
roc_points1 = []
roc_points2 = []
roc_points3 = []
roc_points4 = []
roc_points5 = []
roc_points6 = []


acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = []
#
# y_test = auc_pred['actual']
# predict = auc_pred['prediction']

tresholds = list(np.array(list(range(-100,1000,1)))/800)
for treshold in tresholds:
    tp =[0] * 7; tn = [0] * 7;    fp = [0] * 7;    fn = [0] * 7; tpr = [0] * 7; fpr = [0] * 7;
    for  i in range(len(y_validation)):
        for j in range(len(predict[i])):
            y_pred = predict[i][j]
            if y_pred >= treshold:
                prediction_class = 1
            else:
                prediction_class = 0

            if prediction_class == 1 and y_validation[i] == j+1:
                tp[j]= tp[j] + 1
            elif  y_validation[i] == j+1 and prediction_class ==0:
                fn[j] = fn[j]+1
            elif  y_validation[i] != j+1 and prediction_class == 1:
                fp[j] = fp[j] + 1
            elif  y_validation[i] != j+1 and prediction_class ==0:
                tn[j] = tn[j] + 1
    for k in range(6):
        tpr[k] = tp[k] / (tp[k] + fn[k])
        fpr[k] = fp[k] / (fp[k]+tn[k])
        if k == 0:
            roc_points1.append([tpr[0],fpr[0]])
        elif k == 1:
            roc_points2.append([tpr[1], fpr[1]])
        elif k == 2:
            roc_points3.append([tpr[2], fpr[2]])
        elif k == 3:
            roc_points4.append([tpr[3], fpr[3]])
        elif k == 4:
            roc_points5.append([tpr[4], fpr[4]])
        elif k == 5:
            roc_points6.append([tpr[5], fpr[5]])



df1=pd.DataFrame(roc_points1,columns=["x","y"])
df2=pd.DataFrame(roc_points2,columns=["x","y"])
df3=pd.DataFrame(roc_points3,columns=["x","y"])
df4=pd.DataFrame(roc_points4,columns=["x","y"])
df5=pd.DataFrame(roc_points5,columns=["x","y"])
df6=pd.DataFrame(roc_points6,columns=["x","y"])

cl1 = plt.scatter(df1.y,df1.x)
cl2 = plt.scatter(df2.y,df2.x)
cl3 = plt.scatter(df3.y,df3.x)
cl4 = plt.scatter(df4.y,df4.x)
cl5 = plt.scatter(df5.y,df5.x)
cl6 = plt.scatter(df6.y,df6.x)
plt.plot([0,1])

one = (round(abs(np.trapz(df1.x,df1.y)),4))
two = (round(abs(np.trapz(df2.x,df2.y)),4))
three = (round(abs(np.trapz(df3.x,df3.y)),4))
four = (round(abs(np.trapz(df4.x,df4.y)),4))
five = (round(abs(np.trapz(df5.x,df5.y)),4))
six = (round(abs(np.trapz(df6.x,df6.y)),4))
print(1, one)
print(2, two)
print(3, three)
print(4, four)
print(5, five)
print(6, six)

#

plt.legend((cl1, cl2 , cl3, cl4, cl5, cl6),
           ('Rain=' + str(one),'Fog=' +str(two), 'Precipitation =' +str(three), 'Rain='+str(four), 'Snow='+str(five)
            , 'Storm='+str(six)),
           scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=8)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()
