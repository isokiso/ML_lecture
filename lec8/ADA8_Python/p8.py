import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility


def data_generate(n=50):
    x = np.linspace(-3, 3, n)
    pix = np.pi * x
    y = np.sin(pix) / pix + 0.1 * x + 0.05 * np.random.randn(n)
    return x, y


def sgd(x, y, n, iteration, h=0.3, lr=0.1):
    theta = np.random.randn(n)
    for _ in range(iteration):
        previous_theta = theta.copy()
        sample_i = np.random.randint(0, n)
        k_i = np.exp(-(x - x[sample_i]) ** 2 / (2 * h ** 2))
        gradient = (k_i.dot(theta) - y[sample_i]) * k_i
        theta -= lr * gradient
    return theta


def predict(train_x, test_x, theta, h):
    design_matrix = np.exp(-(train_x[None] - test_x[:, None]) ** 2
                           / (2 * h ** 2))
    return design_matrix.dot(theta)


n, N = 50, 1000
h = 0.3
lr = 0.1

x, y = data_generate()
X = np.linspace(-3, 3, N)
theta = sgd(x, y, n, n*100, h, lr)
prediction = predict(x, X, theta, h)
plt.scatter(x, y)
plt.plot(X, prediction, c='g')
plt.savefig('l8-p8.png')
