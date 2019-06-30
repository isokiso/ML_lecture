import numpy as np
import matplotlib
import scipy.linalg

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def data_generation1(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, np.random.randn(n, 1)],
                          axis=1)

def data_generation2(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, 2 * np.round(
        np.random.rand(n, 1)) - 1 + np.random.randn(n, 1) / 3.], axis=1)

def simirality_mat(x):
    return np.exp(-(np.sum(x**2,axis=1)[:,None] + np.sum(x.T**2, axis=0)[None] - 2*x.dot(x.T)) )


def lpp(x, n_components=1):
    x = x - np.mean(x, axis=0)
    W = simirality_mat(x)
    D = np.diag(W.sum(axis=0))
    L = D - W
    A = x.T.dot(L).dot(x)
    B = x.T.dot(D).dot(x)
    #w, v = np.linalg.eig(B.dot(A).dot(B))
    w, v = scipy.linalg.eig(A, B) #w:固有値 v:固有ベクトル
    return w[:n_components], v[:n_components, :]

def pca(x, n_components=1):
    x = x - np.mean(x, axis=0)
    w, v = np.linalg.eig(x.T.dot(x))
    return w[:n_components], v[:n_components, :]

n = 100
n_components = 1
x = data_generation1(n)
# x = data_generation2(n)
w0, v0 = pca(x, n_components)
w1, v1 = lpp(x, n_components)

plt.xlim(-6., 6.)
plt.ylim(-6., 6.)
plt.plot(x[:, 0], x[:, 1], 'rx')
plt.plot(np.array([-v0[:, 0], v0[:, 0]]) * 9, np.array([-v0[:, 1], v0[:, 1]]) * 9, label="pca")
plt.plot(np.array([-v1[:, 0], v1[:, 0]]) * 9, np.array([-v1[:, 1], v1[:, 1]]) * 9, label='lpp')
plt.legend(loc = 'upper left')
plt.savefig('result1.png')
