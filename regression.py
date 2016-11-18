import csv
import argparse
from sklearn import linear_model
from csv import DictReader, DictWriter
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def reconstruct(path):
    train = list(DictReader(open(path, 'r')))
    mean = []
    for row in train:
        sum = int(row['T1'])+int(row['T2'])+int(row['T3'])+int(row['T4'])+int(row['T5'])+int(row['T6'])+int(row['T7'])+int(row['T8'])+int(row['T9'])+int(row['T10'])
        mean.append(sum/10)

    # print(sorted(train[0].keys())[11:-1])
    fields = ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20"]
    # fields = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"]

    function = []
    ind = 0
    for row in train:
        tmpObj = {}
        for f in fields:
            tmpObj.update({f:row[f]})
        tmpObj.update({'time':mean[int(row['id'])-1]})
        tmpObj.update({'id': ind})
        ind += 1
        function.append(tmpObj)

    fields.insert(0, "time")
    fields.insert(0, "id")
    with open("newData101.csv", 'w') as csvfile:
        writer = DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        ind = 0
        for item in function:
            writer.writerow(dict(item, id=ind))
            ind += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    reconstruct('./data/RegEx_20_10_1_101.csv')
    # reconstruct('./data/MSB1_10_101.csv')

    with open('newData101.csv', 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    #remove column names
    your_list.pop(0)

    train, test = train_test_split(your_list, train_size=0.8)
    X = []
    Y = []
    data = []
    for row in train:
        Y.append(row[1])
        X.append(row[-21:])
    X = np.asmatrix(X).astype(np.float)
    Y = np.asmatrix(Y).astype(np.float).transpose()

    xtest = []
    ytest = []
    for row in test:
        ytest.append(row[1])
        xtest.append(row[-21:])
    xtest = np.asmatrix(xtest).astype(np.float)
    ytest = np.asmatrix(ytest).astype(np.float).transpose()

    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    score = regr.score(xtest, ytest)
    print("Variance Score:", score)
    pred = regr.predict(xtest)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    diff = pred-ytest

    # The mean squared error
    print("Mean squared error: %.2f"
          % np.mean((np.squeeze(np.asarray(diff))) ** 2))

    plt.plot(ytest, pred, color='blue')
    plt.xlabel('Actual Value')
    plt.ylabel('Prediction')

    plt.show()
