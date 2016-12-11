import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# data preprocessing
dataset_list = ['MSB1_10_100', 'RegEx_20_10_1_101', 'RegEx_150_10_2_10011', 'RegEx_200_10_2_101011']
num_func = [11, 21, 151, 201]

df = pd.read_csv("./CSV/" + dataset_list[3] + ".csv")
train, test = train_test_split(df, train_size=0.8)

df_Train = train[['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']]
train['mean'] = (train['T1'] + train['T2'] + train['T3'] + train['T4'] + train['T5'] + train['T6'] + train['T7'] + train['T8'] + train['T9'] + train[
        'T10']) / 10
Mean_Y = np.array(train['mean'].reshape(-1,1))
ytrainl=[]
for i in Mean_Y:
    ytrainl.append(int(i[0]/100))
ytrain = np.asarray(ytrainl)

df_Test = test[['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']]
test['mean'] = (test['T1'] + test['T2'] + test['T3'] + test['T4'] + test['T5'] + test['T6'] + test['T7'] + test['T8'] + test['T9'] + test[
        'T10']) / (10)
Mean_y = np.array(test['mean'].reshape(-1,1))
ytestl=[]
for i in Mean_y:
    ytestl.append(int(i[0]/100))
ytest = np.asarray(ytestl)

func_array = []
for i in range(0, num_func[3]):
    func_array.append('f'+str(i))

df_F = train[func_array]
xtrain = np.array(df_F)

df_F = test[func_array]
xtest = np.array(df_F)

clf = KNeighborsClassifier(3)

#train model
clf.fit(xtrain, ytrain)
score = clf.score(xtest, ytest)
print("Nearest Neighbors", "Score:", score)

pred = clf.predict(xtest)
func_x = []
for i in xtest:
    index = np.nonzero(i)
    func_x.append(index[0][0])

ind = range(len(ytest))

print(classification_report(ytest, pred))

plt.scatter(ind, ytest, color='red', label="Actual Value")
plt.scatter(ind, pred, color='blue', label="Prediction")
plt.title("Nearest Neighbors")
plt.show()
#plt.savefig('./Results/'+item+'/'+name+'.png')


