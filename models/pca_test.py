from models.PCA import PCA
import pandas as pd
from sklearn import preprocessing


def dataset_minmax(dataset):
    minmax = list()
    for column in dataset.columns[0:]:
        col_values = dataset[column].values
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset, minmax):
    i=0
    for column in dataset.columns[0:]:
        dataset[column] = (dataset[column] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        i=i+1


df = pd.read_csv(r'C:\Users\sevki\PycharmProjects\485data\Data\weather_dataset1')
df = df.drop('Unnamed: 0', 1)

# print(df['Severity'].unique())

le = preprocessing.LabelEncoder()

df['Severity'] = df['Severity'].astype(str)
le.fit(df['Severity'])
df['Severity'] = le.transform(df['Severity'])

x_train = df.iloc[:6000,:]
# x_test = df.iloc[4500:,:]

y_train = x_train['Labels']
# y_test = x_test['Labels']
x_train = x_train.drop('Labels', 1)
# x_test = x_test.drop('Labels', 1)

pd.set_option('display.max_columns', None)

p1 = PCA
pcs = p1.fit(x_train,y_train)
p1.plot_2D()
p1.plot_3D()