import numpy as np
import matplotlib
from scipy import linalg

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)


def data_generation(n=1000):
    a = 3. * np.pi * np.random.rand(n)
    x = np.stack(
        [a * np.cos(a), 30. * np.random.random(n), a * np.sin(a)], axis=1)
    return a, x

def euclic_dist(x1, x2):
    xx1 = np.tile(np.sum(x1 * x1, axis=1),(x1.shape[0],1))
    xx2 = np.sum(x2 * x2, axis=1)
    dist = xx1 + xx2 - 2 * x1.dot(x2.T)
    return dist

def kNN(x1,x2, k, dst_type='euclid'):
    if dst_type == 'cosine':
        cosdist = x1.dot(x2.T)
        index = np.argsort(-cosdist, axis=1)[:,:k]
    elif dst_type == 'euclid':
        euc = euclic_dist(x1,x2)
        index = np.argsort(euc, axis=1)[:,1:k+1]
    return index


def LapEig(x, d=2):
    # implement here
    k = 10
    x_norm = (x.T / np.linalg.norm(x, axis=1)).T
    index = kNN(x_norm, x_norm, k, 'euclid')
    n = x.shape[0]
    W = np.zeros([n,n])
    for i in range(n):
        W[i][index[i]] = 1
    W = W + W.T - W*(W.T)
    D = np.diag(np.sum(W, axis=0))
    L = D - W
    eig_val,eig_vec =  linalg.eigh(L,D)
    ret = eig_vec[np.argsort(eig_val.real)][:,1:1+d]
    return ret


def visualize(x, z, a):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=a, marker='o')
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(z[:, 1], z[:, 0], c=a, marker='o')
    fig.savefig('lecture10-h2.png')


n = 1000
a, x = data_generation(n)
z = LapEig(x)
visualize(x, z, a)
