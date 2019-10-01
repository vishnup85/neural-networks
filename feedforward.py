import numpy as np
import matplotlib.pyplot as plt

N = 500

X1 = np.random.randn(N, 2) + np.array([0, -2])
X2 = np.random.randn(N, 2) + np.array([2, 2])
X3 = np.random.randn(N, 2) + np.array([-2, 2])

X = np.vstack([X1, X2, X3])
Y = np.array([0]*N + [1]*N + [2]*N)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)

D = 2
M = 3
K = 3

w1 = np.random.randn(D, M)
b1 = np.random.randn(M)
w2 = np.random.randn(M, K)
b2 = np.random.randn(K)


def forwardprop(X, w1, b1, w2, b2):
    z = np.tanh(X.dot(w1) + b1)
    #z = 1/(1+np.exp(-X.dot(w1) - b1))
    a = z.dot(w2) + b2
    expa = np.exp(a)
    Y = expa/expa.sum(axis=1, keepdims=True)
    return Y


def classification_rate(Y, P):
    return 1-np.mean(Y != P)


Py_given_X = forwardprop(X, w1, b1, w2, b2)
prediction = np.argmax(Py_given_X, axis=1)

print('classification rate is:', classification_rate(Y, prediction))