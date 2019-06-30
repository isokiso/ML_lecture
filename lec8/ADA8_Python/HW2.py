import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)  # set the random seed for reproducibility

def data_generate(n=50):
    x = np.random.randn(n, 3)
    x[:n // 2, 0] -= 15
    x[n // 2:, 0] -= 5
    x[1:3, 0] += 10
    x[:, 2] = 1
    y = np.concatenate((np.ones(n // 2), -np.ones(n // 2)))
    index = np.random.permutation(np.arange(n))
    return x[index], y[index]

def anc(x, y, n, iteration, h=0.3, lr=0.1):
    gamma = .001
    mu = np.random.normal(0,0.1,(3,1))
    sigma = np.random.normal(0,0.1,(3,3))
    for _ in range(iteration):
        index = np.random.randint(0,len(x))
        sample_x = x[index]
        sample_x = sample_x[:,None]
        sample_y = y[index]
        #updata
        beta = sample_x.T.dot(sigma.dot(sample_x)) + gamma
        mu += sample_y * max(0, 1 - mu.T.dot(sample_x) * sample_y) * sigma.dot(sample_x) / beta
        sigma -= sigma.dot(sample_x.dot(sample_x.T.dot(sigma))) / beta
    return mu.flatten(), sigma


def func(x,mu):
    return -mu[0]/mu[1] * x - mu[2]/mu[1]


n, N = 50, 1000
h = 0.3
lr = 0.1

x, y = data_generate()
X = np.linspace(-15, 0, N)
mu, sigma = anc(x, y, n, n*100, h, lr)
prediction = func(X, mu)
axes = plt.gca()
axes.set_xlim(-20,0)
axes.set_ylim(-2,2)
plt.scatter(x[y==1,0], x[y==1,1], c='b', marker='o')
plt.scatter(x[y==-1,0], x[y==-1,1], c='r', marker='x')
plt.plot(X, prediction, c='g')
plt.savefig('result.png')
