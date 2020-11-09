import requests
import numpy as np
from sympy import symbols, diff
import matplotlib.pyplot as plt

class DataSet:

    def __init__(self):
        X = np.loadtxt("../data/project_1/newtons_optimization_x.txt")
        # IMPORTANT
        # print(np.ones((X.shape[0],1)).shape) (99,1)
        # print(np.ones((X.shape[0])).shape) (99,)
        self.X = np.hstack([np.ones((X.shape[0],1)), X])
        self.Y = np.loadtxt("../data/project_1/newtons_optimization_y.txt")

class LogisticRegression():

    def cost_func(self, w, x, y):
        return (w.T @ x - y)**2
    
    # IMPORTANT : for multidimensional setting, we must use hessian for newton optimization instead of derivatives
    # IMPORTANT newton raphson method uses all X and Y values

    # def cost_func_deriv(self, w, x, y):
    #     return (w.T @ x - y) * x * 2

        # u, v, w= symbols("w, x, y")
        # derivative = lambda w, x, y: np.array([float(diff(func(u,v,w),u).subs({u:w, v:x, w:y}))])
        # return derivative(u, x, y)

    def newton_optimizing_method(self, w, X, Y):
        z = Y * (X @ w.T)
        gz = (1/(1+np.exp(-z)))
        cost_func_deriv = np.mean((gz - 1) * Y * X.T, axis=1)

        hessian = np.zeros((X.shape[1],X.shape[1]))
        for i in range(hessian.shape[0]):
            for j in range(hessian.shape[1]):
                hessia_ij = np.mean(gz * (1 - gz) * X[:,i] * X[:,j])
                hessian[i,j] = hessia_ij

        delta = np.linalg.pinv(hessian).T @ cost_func_deriv
        w = w - delta
        return w, abs(sum(delta))

    def main(self):
        self.data_set = DataSet()
        weight = np.zeros(self.data_set.X.shape[1])
        # IMPORTANT don't forget to add the constant term for 2d input
        delta = 1e9
        while delta > 1e-10:
            weight, delta = self.newton_optimizing_method(weight, self.data_set.X, self.data_set.Y)
        self.weight = weight
        
if __name__ == "__main__":
    logistic_regression = LogisticRegression()
    logistic_regression.main()
    data_set = logistic_regression.data_set
    above = data_set.Y > 0
    below = data_set.Y < 0
    plt.scatter(data_set.X[above,1], data_set.X[above,2])    
    plt.scatter(data_set.X[below,1], data_set.X[below,2])    
    x1 = np.array([np.min(data_set.X[:,1]), np.max(data_set.X[:,1])])
    # IMPORTANT w0 + w1x1 + w2x2 = 0
    x2 = (logistic_regression.weight[0] + logistic_regression.weight[1] * x1) / (- logistic_regression.weight[2])
    plt.plot(x1, x2)
    plt.show()

