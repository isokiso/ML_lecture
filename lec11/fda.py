import numpy as np
import matplotlib
from scipy.linalg import eig

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(46)


def generate_data(sample_size=100, pattern='two_cluster'):
    if pattern not in ['two_cluster', 'three_cluster']:
        raise ValueError('Dataset pattern must be one of '
                         '[two_cluster, three_cluster].')
    x = np.random.normal(size=(sample_size, 2))
    if pattern == 'two_cluster':
        x[:sample_size // 2, 0] -= 4
        x[sample_size // 2:, 0] += 4
    else:
        x[:sample_size // 4, 0] -= 4
        x[sample_size // 4:sample_size // 2, 0] += 4
    y = np.ones(sample_size, dtype=np.int64)
    y[sample_size // 2:] = 2
    return x, y


def fda(x, y):
    """Fisher Discriminant Analysis.
    Implement this function

    Returns
    -------
    T : (1, 2) ndarray
        The embedding matrix.
    """
    x -= x.mean(axis=0)
    labels = list(set(y))
    means = np.array([x[y==label].mean(axis=0) for label in labels])

    C = x.T.dot(x)
    S_b = np.array([np.count_nonzero(y == label) * means[ind][:,None].dot(means[ind][None,:]) for ind, label in enumerate(labels)]).sum(axis=0)
    S_w = np.array([(x[y==label] - means[ind]).T.dot((x[y==label] - means[ind])) for ind, label in enumerate(labels)]).sum(axis=0)
    eigvals, eigvecs = eig(S_b, S_w)
    #normalize eigvecs
    eigvecs /= np.linalg.norm(eigvecs,axis=1)
    return eigvecs[0][None,:]

cluster_num = 'three'
def visualize(x, y, T):

    plt.figure(1, (6, 6))
    plt.clf()
    plt.xlim(-7., 7.)
    plt.ylim(-7., 7.)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', label='class-1')
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'rx', label='class-2')
    plt.plot(np.array([-T[:, 0], T[:, 0]]) * 9,
             np.array([-T[:, 1], T[:, 1]]) * 9, 'k-')
    plt.legend()
    plt.savefig('result_' + cluster_num + '.png')


sample_size = 100
# x, y = generate_data(sample_size=sample_size, pattern='two_cluster')
x, y = generate_data(sample_size=sample_size, pattern= cluster_num + '_cluster')
T = fda(x, y)
visualize(x, y, T)
