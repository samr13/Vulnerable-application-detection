import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# df = pd.read_csv("./CSV/RegEx_20_10_1_101.csv")
df = pd.read_csv("./CSV/RegEx_150_10_2_10011.csv")
# df = pd.read_csv("./CSV/RegEx_200_10_2_101011.csv")
# df = pd.read_csv("./CSV/MSB1_10_100.csv")

train, test = train_test_split(df, train_size=0.8)

# process training set
train_T = train[['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']]
train['mean'] = (train['T1'] + train['T2'] + train['T3'] + train['T4'] + train['T5'] + train['T6'] + train['T7'] +
                 train['T8'] + train['T9'] + train[
    'T10']) / 10

train_Mean = np.array(train['mean'].reshape(-1,1))
m_list=[]
for i in train_Mean:
    m_list.append(i[0])
Y = np.asarray(m_list)

func_array = []
for i in range(0, 151):
    func_array.append('f'+str(i))
train_F = train[func_array]
X = np.array(train_F)


# process test set
test_T = test[['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']]
test['mean'] = (test['T1'] + test['T2'] + test['T3'] + test['T4'] + test['T5'] + test['T6'] + test['T7'] +
                 test['T8'] + test['T9'] + test[
    'T10']) / 10
test_Mean = np.array(test['mean'].reshape(-1,1))
m_list=[]
for i in test_Mean:
    m_list.append(i[0])
y_true = np.asarray(m_list)

func_array = []
for i in range(0, 151):
    func_array.append('f'+str(i))
test_F = test[func_array]
xtest = np.array(test_F)

regr = linear_model.LinearRegression()
regr.fit(X, Y)
score = regr.score(xtest, y_true)
print("Variance Score:", score)
pred = regr.predict(xtest)
# The coefficients
print('Coefficients: \n', regr.coef_)
print('Bias: ', regr.intercept_)

# Print Formula
formula = ""
index = 0
for i in regr.coef_:
    if i>=0:
        formula += '+'+str(i)+'f'+str(index)
    else:
        formula += str(i)+'f'+str(index)
    index += 1

print('formula', str(regr.intercept_)+formula)
# diff = pred - y_true

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_true, pred))

from collections import defaultdict
d = defaultdict(list)

func_x = []
for i in xtest:
    index = np.nonzero(i)
    func_x.append(sum(index[0]))
    d[sum(index[0])] = index[0]

for key, value in d.iteritems():
    print('label: ', key,'function index:', value)

plt.scatter(func_x, pred, color='red', label="Prediction")
plt.scatter(func_x, y_true, color='blue', marker='*', label="Actual Value")

# plt.plot(ytest, pred, color='blue')
plt.xlabel('Labels')
plt.ylabel('Run Time')
# plt.title('RegEx_20_10_1_101 dataset')
# plt.title("RegEx_150_10_2_10011")
# plt.title("RegEx_200_10_2_101011")
plt.title("MSB1_10_100")

plt.legend(loc=4)

# plt.show()
