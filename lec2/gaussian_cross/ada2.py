import numpy as np
import matplotlib.pyplot as plt

def main():
 np.random.seed(0)
 l =0.1 #lambda
 h = 0.1 #band
 knum = 50 #cross
 hh = 2*h**2
 n = 50
 N = 1000
 x = np.linspace(-3, 3, n)
 X = np.linspace(-3, 3, N)
 pix = np.pi * x
 y = np.sin(pix)/pix+0.1*x+0.2*np.random.randn(n)

 order = np.random.permutation(n)
 index = np.arange(0, n, knum)

 errors = np.zeros(shape=int(n/knum,))

 for e, i in enumerate(index):
   # optimize
   x_b = np.hstack((x[order[:i]], x[order[i+knum:]]))
   y_b = np.hstack((y[order[:i]], y[order[i+knum:]]))
   k = np.exp(-(np.square(x_b-x_b[:,None])/hh))
   t = (np.linalg.inv(k.dot(k)+l*np.eye(len(x_b))).dot(k)).dot(y_b)

   # validation
   x_v = x[order[i:i+knum]]
   y_v = y[order[i:i+knum]]
   k_v = np.exp(-(np.square(x_v-x_b[:,None])/hh))
   y_b = k_v.T.dot(t)

   # error
   error = np.sum(np.square(y_v-y_b)) / knum
   print(e, error)
   errors[e] = error

 print("MEAN: ", np.mean(errors))


 k = np.exp(-(np.square(x-x[:,None])/hh))
 K = np.exp(-(np.square(X-x[:,None])/hh))
 t = (np.linalg.inv(k.dot(k)+l*np.eye(len(x))).dot(k)).dot(y)
 y_b = K.T.dot(t)
 plt.plot(x,y, 'bo')
 plt.plot(X, np.sin(np.pi*X)/(np.pi*X)+0.1*X,'r-')
 plt.plot(X, y_b,'g-')
 plt.title("λ={}; h={}; error={}".format(l, h, np.mean(errors)))
 plt.savefig("graph9.png")


if __name__ == '__main__':
 main()
