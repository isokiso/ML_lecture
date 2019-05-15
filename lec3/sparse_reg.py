from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
from fun_tripolybase import generate_sample

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility

#gaussian kernel
def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

h = 0.3
l = 0.1
k = calc_design_matrix(x,x,h)

def update_theta(u,z):
    return np.linalg.inv(k.T.dot(k) + np.identity(len(x))).dot(k.T.dot(y) - u.T + z)
def update_z(theta, u):
    return np.maximum(0,theta + u.T - l) + np.minimum(0,theta + u.T + l)
def update_u(u,theta,z):
    return u+theta-z

#initialize params
theta = np.random.randn(sample_size)
u = np.random.randn(sample_size)
z = np.random.randn(sample_size)

#theta hist
plt.clf()
plt.xlabel('theta')
plt.ylabel('num')
plt.hist(theta, bins = 25,align = 'mid')
plt.savefig('theta_0.png')

update_num = 100
error = np.zeros(update_num)
for i in range(update_num):
    theta = update_theta(u,z)
    z = update_z(theta,u)
    u = update_u(u,theta,z)

    pred = k.T.dot(theta)
    error[i] = np.linalg.norm(y - pred)

    if(i == 9):
        #theta hist
        plt.clf()
        plt.xlabel('theta')
        plt.ylabel('num')
        plt.hist(theta, bins = 25,align = 'mid')
        plt.savefig('theta_10.png')

# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=5000)
K = calc_design_matrix(x, X, h)
k = calc_design_matrix(x, x, h)
prediction = K.dot(theta)

# visualization
plt.clf()
plt.scatter(x, y, c='green', marker='o')
plt.plot(X, prediction)
plt.savefig('fitting.png')

#error
plt.clf()
plt.xlabel('trial')
plt.ylabel('error')
plt.plot(range(update_num),error)
plt.savefig('error.png')

#theta hist
plt.clf()
plt.xlabel('theta')
plt.ylabel('num')
plt.hist(theta, bins = 25,align = 'mid')
plt.savefig('theta.png')
