import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
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
print(df.head())
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
    temp_df =  original_df[(original_df['cluster_group']==float(i))]
    survival_cluster = temp_df[ (temp_df['survived']==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate
print(survival_rates)
    







