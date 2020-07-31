from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

try:
    xrange
except NameError:
    xrange = range

# IMPORTANT logistic regression, maximum likelihood method can not handle perfectly seperable data
# since |ceta * x| must be infinity for h(x) to equal to 1 or 0
# IMPORTANT svm will not be affect be scaling with ceta(i.e. w and b) since we are trying to maximise the geometric margin not the functional margin

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        norm = np.linalg.norm(prev_theta - theta)
        if i % 1000 == 0:
            plt.scatter(X[Y>0,1], X[Y>0,2], label="dataset_+")
            plt.scatter(X[Y<0,1], X[Y<0,2], label="dataset_-")
            plt.legend()
            plt.plot(np.arange(0, 1, 0.1), (- theta[0] - theta[1] * np.arange(0, 1, 0.1)) / theta[2])
            plt.show()
        if i % 10000 == 0:
            print('Finished {0} iterations; Diff theta: {1}; theta: {2}; Grad: {3}'.format(
                i, norm, theta, grad))
        if norm < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = load_data('./stanford_ml/data_a.txt')
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = load_data('./stanford_ml/data_b.txt')
    # logistic_regression(Xb, Yb)


    return

if __name__ == '__main__':
    main()