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


def LapEig(x, d=2):
    # implement here
    pass


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
