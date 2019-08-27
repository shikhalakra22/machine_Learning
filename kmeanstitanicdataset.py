import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


def handle_data(df):
    columns = df.columns.values

    for column in columns:
        tdv = {}
        def convert_to_int(val):
            return tdv[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            cc = df[column].values.tolist()
            ue = set(cc)
            x = 0
            for u in ue:
                if u not in tdv:
                   tdv[u] = x
                   x+=1

            df[column] = list(map(convert_to_int, df[column]))
    return df
df = handle_data(df)
#print(df.head())
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

c = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        c = c+1
print(c/len(X))
