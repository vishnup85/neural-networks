import numpy as np
import matplotlib.pyplot as plt


def feedforward(X, w1, b1, w2, b2):
    z = 1/(1+np.exp(-X.dot(w1)-b1))
    a = z.dot(w2) + b2
    expa = np.exp(a)
    Y = expa/expa.sum(axis=1, keepdims=True)
    return Y, z


def classification_rate(Y, P):
    return 1-np.mean(Y != P)


# def derivative_w2(Z, T, Y):
#     N, K = T.shape
#     M = Z.shape[1]
#
#     ret1 = np.zeros((M, K))
#     for n in range(N):
#         for m in range(M):
#             for k in range(K):
#                 ret1[m, k] += (T[n, k] - Y[n, k])*Z[n, m]
#
#     return ret1

def derivative_w2(Z, T, Y):
    return Z.T.dot(T-Y)


# def derivative_b2(T, Y):
#     N, K = T.shape
#     ret2 = np.zeros(K)
#     for n in range(N):
#         for k in range(K):
#             ret2[k] = np.sum(T[n, k] - Y[n, k], axis=0)
#     return ret2

def derivative_b2(T, Y):
     return (T-Y).sum(axis=0)


# def derivative_w1(X, Z, T, Y, w2):
#     N, K = T.shape
#     M = Z.shape[1]
#     D = X.shape[1]
#     ret1 = np.zeros((D, M))
#     for n in range(N):
#         for k in range(K):
#             for m in range(M):
#                 for d in range(D):
#                     ret1[d, m] += (T[n, k] - Y[n, k])*(w2[m, k])*Z[n, m]*(1 - Z[n, m])*X[n, d]
#     return ret1
def derivative_w1(X, Z, T, Y, w2):
    dz = (T-Y).dot(w2.T)*Z*(1-Z)
    return X.T.dot(dz)


# def derivative_b1(Z, T, Y, w2):
#     N, K = T.shape
#     M = Z.shape[1]
#     ret2 = np.zeros(M)
#     for n in range(N):
#         for m in range(M):
#             for k in range(K):
#                 ret2[m] += np.sum((T[n, k] - Y[n, k])*(w2[m, k])*Z[n, m]*(1 - Z[n, m]), axis=0)
#     return ret2
def derivative_b1(Z, T, Y, w2):
    return ((T-Y).dot(w2.T)*Z*(1-Z)).sum(axis=0)


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()


def main():
    Nclass = 500
    D = 2
    M = 3
    K = 3
    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])

    X = np.vstack([X1, X2, X3])
    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100)
    N = len(Y)

    # convert the targets into indicator matrix
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1


    # weight initialization
    w1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    w2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    # backpropagation
    learning_rate = 10e-07
    costs = []
    for epoch in range(100000):
        output, hidden = feedforward(X, w1, b1, w2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            prediction = np.argmax(output, axis=1)
            class_rate = classification_rate(Y, prediction)
            costs.append(c)
            print("cost: ", c, "classifcation rate: ", class_rate)

        # gradient ascent
        w2 += learning_rate*derivative_w2(hidden, T, output)
        b2 += learning_rate*derivative_b2(T, output)
        w1 += learning_rate*derivative_w1(X, hidden, T, output, w2)
        b1 += learning_rate*derivative_b1(hidden, T, output, w2)

    plt.plot(costs)
    plt.show()


if __name__=='__main__':
    main()