import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2.
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = 0
    y[n_positive:] = 1

    return x, y

def cal_norm(x1, x2):
    return np.mean(np.linalg.norm(x1[None,:,:] - x2[:,None,:], axis=2))

def cwls(train_x, train_y, test_x):
    # weight
    xp = train_x[train_y==1]
    xn = train_x[train_y==0]
    Apn = cal_norm(xp, xn)
    Ann = cal_norm(xn, xn)
    App = cal_norm(xp, xp)
    bp = cal_norm(test_x, xp)
    bn = cal_norm(test_x, xn)
    pi = np.minimum(1, np.maximum(0, (Apn - Ann - bp + bn) / (2*Apn - App - Ann)))

    p_train_p = xp.shape[0] / train_x.shape[0]
    p_train_n = 1-p_train_p
    w = np.select([train_y==1, train_y==0],[pi / p_train_p, (1-pi) / (1-p_train_p)])
    W = np.diag(w)

    Phil = np.hstack((train_x, np.ones(train_x.shape[0])[:,np.newaxis]))
    d = Phil.shape[1]
    I = np.eye(d)
    l=.1
    train_y = np.eye(2)[train_y.astype('int32')]
    theta = np.linalg.inv(Phil.T.dot(W).dot(Phil) + l*I).dot(Phil.T).dot(W).dot(train_y)
    theta = theta[:,0] - theta[:,1]

    return theta


def visualize(train_x, train_y, test_x, test_y, theta):
    for x, y, name in [(train_x, train_y, 'train'), (test_x, test_y, 'test')]:
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.xlim(-5., 5.)
        plt.ylim(-7., 7.)
        lin = np.array([-5., 5.])
        plt.plot(lin, -(theta[2] + lin * theta[0]) / theta[1])
        plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1],
                    marker='$O$', c='blue')
        plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1],
                    marker='$X$', c='red')
        plt.savefig('lecture9-h3-{}.png'.format(name))


train_x, train_y = generate_data(n_total=100, n_positive=90)
eval_x, eval_y = generate_data(n_total=100, n_positive=10)
theta = cwls(train_x, train_y, eval_x)
visualize(train_x, train_y, eval_x, eval_y, theta)
