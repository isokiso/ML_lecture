import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)

sample_size = 90
n_class = 3
def generate_data(sample_size, n_class):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y

def build_design_mat(x1, x2, bandwidth):
    ret = np.exp(-(x1[None]-x2[:,None]) ** 2 / (2 * bandwidth ** 2))
    return ret


def ls_prob_clustering(x, y, h, l):
    sample_size = len(x)
    Phi = build_design_mat(x,x,l)
    pi = np.tile(np.identity(n_class),(sample_size // n_class, 1))
    theta = np.linalg.inv(Phi.dot(Phi) + l * np.identity(sample_size)).dot(Phi).dot(pi)
    return theta


def visualize(x, y, theta, h):
    X = np.linspace(-5., 5., num=100)
    K = build_design_mat(x,X,h)

    plt.clf()
    plt.xlim(-5, 5)
    plt.ylim(-.3, 1.8)
    unnormalized_prob = K.dot(theta)
    unnormalized_prob[unnormalized_prob < 0] = 0
    prob = unnormalized_prob / unnormalized_prob.sum(axis=1)[:,None]

    plt.plot(X, prob[:, 0], c='blue')
    plt.plot(X, prob[:, 1], c='red')
    plt.plot(X, prob[:, 2], c='green')

    plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')

    plt.savefig('result.png')


h=1.
l=.1
x, y = generate_data(sample_size=90, n_class=3)
theta = ls_prob_clustering(x, y, h, l)
visualize(x, y, theta, h)
