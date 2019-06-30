import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
import collections

np.random.seed(1)


def generate_train_data():
    x = np.ndarray(shape=(0, 256))
    y = np.ndarray(shape=(0, 10))
    for i in range(10):
        _x = pd.read_csv("digit_train{}.csv".format(i), header=None)
        x = np.concatenate([x, _x])
        _y = -np.ones(shape=(_x.shape[0],10))
        _y[:,i] = 1
        y = np.concatenate([y, _y])
    print(x.shape)
    return x, y


def generate_test_data():
    x = np.ndarray(shape=(0, 256))
    y = np.ndarray(shape=(0, 10))
    for i in range(10):
        _x = pd.read_csv("digit_test{}.csv".format(i), header=None)
        x = np.concatenate([x, _x])
        _y = -np.ones(shape=(_x.shape[0],10))
        _y[:,i] = 1
        y = np.concatenate([y, _y])
    return x,y


def build_design_mat(x1, x2, bandwidth):
    x11=np.sum(x1*x1, axis=1)
    x22=np.sum(x2*x2, axis=1)
    return np.exp(-(x11+x22[:,None]-2*x2.dot(x1.T))/ (2 * bandwidth ** 2))

def optimize_param(design_mat, y, regularizer):
    return np.linalg.solve(
        design_mat.T.dot(design_mat) + regularizer * np.identity(len(y)),
        design_mat.T.dot(y))


def predict(train_data, test_data, theta):
    return build_design_mat(train_data, test_data, 10.).dot(theta)


def build_confusion_matrix(train_data, test_data, theta):
    confusion_matrix = np.zeros((10, 10), dtype=np.int64)
    for i in range(10):
        prediction = np.argmax(predict(train_data, test_data[200*i:200*(i+1)], theta),axis=1)
        for j in range(10):
            confusion_matrix[i][j] = len(np.where(prediction==j)[0])
    return confusion_matrix

os.chdir("../digit")
print('Loading Training Data')
x, y = generate_train_data()
print('Start Training')
#h=10. lambda=1.
design_mat = build_design_mat(x, x, 10.)
theta = optimize_param(design_mat, y, 1.)

print('Loading Test Data')
test_x, test_y = generate_test_data()

confusion_matrix = build_confusion_matrix(x, test_x, theta)
print('confusion matrix:')
print(confusion_matrix)
print('accuracy')
print(confusion_matrix.trace()/test_x.shape[0])
