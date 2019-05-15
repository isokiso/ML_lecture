from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility


def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

hopt,lopt = 0, 0
diff_min=100
k_num = 50 #k分割
bc = sample_size/k_num #バッチサイズ

for h in range(1,11):
    h *= 0.1
    for l in range(1,11):
        l *= 0.1
        diff_sum = 0
        for i in range(k_num):#(bc)個除いて最小二乗法
            idx = [int(i*bc),int(i*bc+bc)] #iからi+bcまでをテスト用にする
            #eliminate sample
            elim = range(idx[0],idx[1])
            x_del = np.delete(x,elim)
            y_del = np.delete(y,elim)
            # calculate design matrix
            k = calc_design_matrix(x_del, x_del, h)
            # solve the least square problem
            theta = np.linalg.solve(
                k.T.dot(k) + l * np.identity(len(k)),
                k.T.dot(y_del[:, None]))

            #テスト誤差
            y_test = y[idx[0]:idx[1]]
            diff_sum += (np.linalg.norm(k.dot(theta)[idx[0]:idx[1]] - y_test.T))**2 / bc
        diff_ave = diff_sum/(k_num*1.0)
        if diff_ave < diff_min:
            diff_min = diff_ave
            hopt = h
            lopt = l
        if(h == 0.1 and l ==0.1):
            print('h = {hopt}\nlambda = {lopt}\ndiff_ave = {diff}'.format(hopt=hopt,lopt=lopt,diff=diff_min))
print('h = {hopt}\nlambda = {lopt}\ndiff_ave = {diff}'.format(hopt=hopt,lopt=lopt,diff=diff_min))

# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=5000)
K = calc_design_matrix(x, X, hopt)
k = calc_design_matrix(x, x, hopt)
theta = np.linalg.solve(
    k.T.dot(k) + lopt * np.identity(len(k)),
    k.T.dot(y[:, None]))
prediction = K.dot(theta)

# visualization
plt.clf()
plt.scatter(x, y, c='green', marker='o')
plt.plot(X, prediction)
plt.savefig('fitting.png')
