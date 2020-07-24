import cvxopt 
import numpy as np
import matplotlib.pyplot as plt

def f_original(x):
    return 3*x[:,0]**2 + x[:,1]**2 + 2*x[:,0]*x[:,1] + x[:,0] + 6*x[:,1] + 2

def f_matrix(x):
    P = np.array([[1.5, 1],
                  [1, 0.5]])
    q = np.array([[1], [6]])
    # IMPORTANT numpy array.T is the transpose form
    return 1/2 * x.T @ P @ x + q.T @ x

def constraints_satisfied(x):
    G = np.array([
        [-2, 3],
        [-1, 0],
        [0, -1],
    ])
    h = np.array([[4], [0], [0]])
    return G @ x <= h


def plot_contour(function_, range_=[-1,1]): # IMPORTANT, don't give any python method as names for variabls

    xx, yy = np.linspace(range_[0],range_[1],100), np.linspace(range_[0],range_[1],100) # IMPORTANT, xx, yy === x1, x2
    XX, YY = np.meshgrid(xx, yy) # IMPORTANT, XX = [[-2...20]..], YY = [[-5...-5]...[5...5]]
    ZZ_matrix = np.zeros(XX.shape)
    constraints = np.zeros(XX.shape)
    
    for row in range(100): # IMPORTANT, np array.shape[0] ==  len(array), nparray.shape[1] == len(array[0])
        for col in range(XX.shape[1]):
            x = np.array([[XX[row,col]], [YY[row,col]]])
            ZZ_matrix[row,col] = function_(x)
            if np.all(constraints_satisfied(x)):
                constraints[row,col] = 1

    plt.xlabel("x")
    plt.ylabel("y")
    plt.contourf(XX, YY, ZZ_matrix, 100)
    plt.contour(XX, YY, constraints)

    plt.show()

plot_contour(function_)