from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt

# data preprocessing
df = pd.read_csv("./CSV/MSB1_10_100.csv")
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

df_F = train[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']]
xtrain = np.array(df_F)

df_F = test[['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']]
xtest = np.array(df_F)

#classifiers
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

#train model
i = 1
for name, clf in zip(names, classifiers):
    clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)
    print(name, "Score:", score)
    pred = clf.predict(xtest)
    func_x = []
    for i in xtest:
        index = np.nonzero(i)
        func_x.append(index[0][0])

    ind = range(len(ytest))


    plt.scatter(ind, ytest, color='red', label="Actual Value")
    plt.scatter(ind, pred, color='blue', label="Prediction")
    plt.title(name)
    i += 1
    # plt.show()
    plt.savefig('./Results/'+name+'.png')

