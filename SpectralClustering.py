import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import matplotlib.pyplot as plt

# df = pd.read_csv("./CSV/RegEx_20_10_1_101.csv")
# df = pd.read_csv("./CSV/RegEx_150_10_2_10011.csv")
df = pd.read_csv("./CSV/RegEx_200_10_2_101011.csv")
# df = pd.read_csv("./CSV/MSB1_10_100.csv")


df_T = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']]
df['mean'] = (df['T1'] + df['T2'] + df['T3'] + df['T4'] + df['T5'] + df['T6'] + df['T7'] + df['T8'] + df['T9'] + df[
    'T10']) / 10
Mean = np.array(df['mean'].reshape(-1,1))
list=[]
for i in Mean:
    list.append(i[0])
labels_true = np.asarray(list)

func_array = []
for i in range(0, 201):
    func_array.append('f'+str(i))
df_F = df[func_array]
X = np.array(df_F)

# X = StandardScaler().fit_transform(X)

spectral = cluster.SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity='nearest_neighbors')

sc = spectral.fit(X)
y_pred = sc.labels_.astype(np.int)


colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)