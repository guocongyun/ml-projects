import numpy as np
from sklearn.linear_model import Perceptron
import cvxopt
import matplotlib.pyplot as plt

class Dataset:

    def __init__(self):
        self.X = None
        self.Y = None
        self.target_function = None

    def generate_data(self, n=10):

        split_data = False
        while not split_data:        
            self.X, self.Y  = [], []
            for _ in range(n): # don't use np append in a loop as it's slow

                x1 , x2 = self.rand_point(2)
                x = np.array([1, x1, x2])
                self.X.append(x)

            self.X = np.array(self.X)
            self.Y = np.array(self.target_function(self.X))
            if sum(self.Y > 0) != 0 and sum(self.Y < 0) != 0: split_data = True

        return self.X, self.Y

    def create_target_function(self):
        x0, y0, x1, y1 = self.rand_point(4)
        slope = (y1-y0)/(x1-x0)
        constant = y0 - slope * x0
        self.target_function = lambda X: np.sign(X[:, 0]*constant + X[:, 1]*slope - X[:, 2]) 
        # IMPORTANT, np.array[:,1] != np.array[1][:] == np.array[:][1]
        # since array[:] = array
        # however np.array[a,b] == np.array[a][b]

    def rand_point(self,n):
        return np.random.uniform(-1,1,size=n)


#%%

def linear_kernel(x1,x2):
    return x1.T @ x2

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

#%%

class SVM():

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        self.weights = 0
        self.alphas = []
        self.bias = 0

    def fit(self, X, y):

        n_samples, n_paramters = X.shape
        self.weights = np.zeros(n_paramters)

        K = np.zeros((n_samples,n_samples)) # IMPORTANT, if all y are +1 or -1 then no support vector will be found
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i],X[j])

        P =  cvxopt.matrix(np.outer(y,y) * K)
        q =  cvxopt.matrix(-np.ones(n_samples))

        A = cvxopt.matrix(y, (1,n_samples)) # this transform matrix([1,-1,1])== [[1],[-1],[1]] into one row:[1,-1,1]
        b = cvxopt.matrix(0.0)

        if self.C is None:
            # IMPORTANT: G.shape == n_samples,n_samples
            # h.shape == n_samples,1
            # self.alphas.shape == n_samples,1
            # print(np.shape(h))
            G = cvxopt.matrix(-np.diag(np.ones(n_samples)))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            # IMPORTANT: G.shape == 2*n_samples,n_samples(the upper is the first condition)
            # h.shape == 2*n_samples,1
            # self.alphas.shape == n_samples,1
            upper_bound = np.diag(-np.ones(n_samples))
            lower_bound = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((lower_bound,upper_bound)))
            upper_bound = np.zeros(n_samples)
            lower_bound = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((lower_bound,upper_bound)))

        cvxopt.solvers.options["show_progress"] = False # this hide the print messge of cvxopt
        qp_sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(qp_sol["x"])
        if self.C is None:
            self.sv = ((self.alphas > 1e-5)).flatten()
        else:
            self.sv = ((self.alphas > 1e-5) * (self.alphas < self.C)).flatten()
        # IMPORTANT here sv = ((self.alphas > 1e-5) * (self.alphas < self.C)).flatten() the * means and
        # .flatten() changes [[1].[1].[1]] into [1,1,1]

        for sv_num in range(len(X[self.sv])):
            self.weights += self.alphas[self.sv][sv_num] * X[self.sv][sv_num] * y[self.sv][sv_num]

        # IMPORTANT y[sv, np.newaxis] == (y = y[sv] and y[:, np.newaxis])
        # IMPORTANT, if list l=[True, False..] K[l][l] != K[l, l] since K[l][l] leads to err
        # K[sv,sv][0] === X[sv][0].T @ X[sv][0] where K[i,j] == X[i].T @ X[j] 
        # for sv_num in range(len(X[sv])): # IMPORTANT, X[sv] ==[[1,2,3],[1,2,3]...True]!= X[num]

        for sv_num in range(len(X[self.sv])):
            # self.bias += np.sum(y[sv][sv_num] - (X[sv].T @ (self.alphas[sv] * y[sv,np.newaxis]).flatten()) @ X[sv][sv_num])
            self.bias += np.sum(y[self.sv][sv_num] - self.weights @ X[self.sv][sv_num])
        try:
            self.bias /= len(X[self.sv])
        except:
            pass
        
    def predict(self, X, classify=True):
        if classify: self.prediction = np.sign(self.weights @ X.T + self.bias)
        else:self.prediction = self.weights @ X.T + self.bias
            
        return self.prediction

#%%

def plot_points(X, y):
    below = np.where(y < 0)
    above = np.where(y > 0)
    plt.scatter(X[below,1], X[below,2]) # IMPORTANT, different scatter methods uses different colors
    plt.scatter(X[above,1], X[above,2])

def plot_contour(function_, dataset, range_=[-1,1]): # IMPORTANT, don't give any python method as names for variabls

    xx, yy = np.linspace(range_[0],range_[1],100), np.linspace(range_[0],range_[1],100) # IMPORTANT, xx, yy === x1, x2
    # linspace create 100 element list
    XX, YY = np.meshgrid(xx, yy) # IMPORTANT, XX = [[-2...20]..], YY = [[-5...-5]...[5...5]]
    # mshgrid create 100x100 element matrix
    ZZ = np.zeros(XX.shape)

    # 1) creating using matrix multiplications
    ZZ = function_.predict(np.array([1,XX, YY]),False)
    print(np.array([1,XX, YY]))
    print(ZZ)

    # 2) creating ZZ list using for loops
    # for row in range(XX.shape[0]): # IMPORTANT, np array.shape[0] ==  len(array), nparray.shape[1] == len(array[0])
    #     for col in range(XX.shape[1]):
    #         x = np.array([[1,XX[row,col], YY[row,col]]])
    #         ZZ[row,col] = function_.predict(x,False)
            # if np.all(constraints_satisfied(x)):
            #     constraints[row,col] = 1

    # 3) creating ZZ list using Xplot new list, 1)==2)==3)
    # Xplot = np.zeros((Nplot**2, 3))
    # Xplot[:,0] = 1
    # Xplot[:,1] = XX.reshape(-1)
    # Xplot[:,2] = YY.reshape(-1)
    # yplot = function_.predict(Xplot[:],False)
    # yplot = np.sign(yplot[:] + function_.bias)
    # ZZ = yplot.reshape(XX.shape)
    plt.gca().set_aspect(1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.contourf(XX, YY, ZZ, [-1,0,1]) # the last parameter is the range of the 2 level, from -1 to 0 and from 0 to 1

    plot_points(dataset.X,dataset.Y)
    # plt.legend(["-1", "+1"]) # ISSUE, having legend cause contour level to be distorted
    plt.show()

#%%
RUNS = 100
TRAIN_DATA_NUM = 2
TEST_DATA_NUM = 1000

#%%

if __name__ == "__main__":
    dataset = Dataset()
    svm_erro = 0
    sv_vs_pla = 0
    sv_num = 0
    pla_erro = 0
    for _ in range(RUNS):
        dataset.create_target_function()
        dataset.generate_data(TRAIN_DATA_NUM)
        pla = Perceptron(max_iter=1000).fit(dataset.X,dataset.Y) # if fit_intercept=False the learning algorithm will force y intercept at the origin 0
        svm = SVM()
        svm.fit(dataset.X,dataset.Y)

        plot_contour(svm,dataset)

        dataset.generate_data(TEST_DATA_NUM)
        svm_predict = svm.predict(dataset.X)
        pla_predict = pla.predict(dataset.X)

        sv_num += len(svm.alphas[svm.sv])
        svm_erro += sum(svm_predict!=dataset.Y)/TEST_DATA_NUM
        pla_erro += sum(pla_predict!=dataset.Y)/TEST_DATA_NUM
        if sum(svm_predict!=dataset.Y)/TEST_DATA_NUM < sum(pla_predict!=dataset.Y)/TEST_DATA_NUM:
            sv_vs_pla += 1

#%%

    # IMPORTANT print(f"") means print formatted strings
    print(f"num of SVM is : {sv_num/RUNS}")
    print(f"SVM is {sv_vs_pla/RUNS*100}% better than PLA") # IMPORTANT, percentage better doesn't mean percentage less error
    print(f"out sample error for SVM is : {svm_erro/RUNS}")
    print(f"out sample error for PLA is : {pla_erro/RUNS}")

# A = np.array([2,0,1,8])
# A[np.newaxis,:] = np.array([[2,0,1,8]])
# A[:, np.newaxis] = np.array([[2],[0],[1],[8]])

# IMPORTANT np.reshape(-1) means to reshape array to a unknown dimension
# i.e. [1,2].reshape(-1,1) => [[1],[2]].shape = (2,1)



# %%
