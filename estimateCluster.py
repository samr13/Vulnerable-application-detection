import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

# Comment here to change dataset, also the lenth of functions below
df = pd.read_csv("./CSV/RegEx_20_10_1_101.csv")
# df = pd.read_csv("./CSV/RegEx_150_10_2_10011.csv")
# df = pd.read_csv("./CSV/RegEx_200_10_2_101011.csv")
# df = pd.read_csv("./CSV/MSB1_10_100.csv")


df_T = df[['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']]
df['mean'] = (df['T1'] + df['T2'] + df['T3'] + df['T4'] + df['T5'] + df['T6'] + df['T7'] + df['T8'] + df['T9'] + df[
    'T10']) / 10
Mean = np.array(df['mean'].reshape(-1,1))
m_list=[]
for i in Mean:
    m_list.append(i[0])
labels_true = np.asarray(m_list)

# At here change the length according to different dataset [21, 151, 201, 11]
func_array = []
for i in range(0, 21):
    func_array.append('f'+str(i))
df_F = df[func_array]
X = np.array(df_F)
# print(df_F)

# A = csr_matrix(X)
# X = A.todense()
# print('check',type(X))

#X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
'''''
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)
'''''
colors = plt.cm.Spectral(np.linspace(0, 1, len(Mean)))

from collections import defaultdict
d = defaultdict(list)
label_func = defaultdict(list)
hash_arr = defaultdict(set)
i=0
for col in colors:
    d[labels[i]].append(Mean[i][0])
    label_func[labels[i]].append(np.nonzero(X[i])[0])
    if labels[i]!=-1:
        plt.plot(labels[i], Mean[i][0], 'o', markerfacecolor=col, markersize=10)
    i += 1

minVal = []
maxVal = []
ind = []
diff = []
for key, value in d.iteritems():
    if key!=-1:
        minVal.append(min(value))
        maxVal.append(max(value))
        diff.append(max(value)-min(value))
        # print('max',max(value), 'min', min(value))
        ind.append(key)
plt.xlabel('Labels')
plt.ylabel('Average Run Time')
plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.show()


plt.plot(ind, minVal, 'r')
plt.plot(ind, maxVal, 'b')
plt.plot(ind, diff, 'c')
plt.xlabel('Labels')
plt.ylabel('Average Run Time')
plt.title('Min/Max Run Time of Each Label')
plt.show()

for key, value in label_func.iteritems():
    print('label', key, 'functions', value[0])
