import requests
import numpy as np
import scipy as sp
import random
from sklearn import svm, model_selection, linear_model
import cvxopt
import seaborn as sb
import matplotlib.pyplot as plt

#%%

class DataSet:

    def __init__(self):
        self.X = []
        self.Y = []
        self.Data = []


    def generate_points(self, num):
        self.X = np.array([np.ones(num),np.random.uniform(-1,1,num),np.random.uniform(-1,1,num)]).T
        self.Y = self.target_function(self.X)
        self.Data = np.vstack((self.X.T,self.Y.T)).T

        # IMPORTANT np.vstack((A.T B.T)).T == np.column_stack((A B)) == np.concatenate((A.T B.T), axis=0).T
        # IMPORTNAT axis=0 is rows axis=1 is columns

    def target_function(self, X, classify = True):
        if classify: return np.sign(X[:,2] - X[:,1] + 0.25*np.sin(np.pi*X[:,1]))
        if not classify: return X[:,2] - X[:,1] + 0.25*np.sin(np.pi*X[:,1])

#%%

def plot_points(data, plot):
    above = (data[:,3] > 0)
    below = (data[:,3] < 0)
    # print(data_set)
    plot.scatter(data[above,1],data[above,2])
    plot.scatter(data[below,1],data[below,2])
    if (data[0][0] == 0): plot.scatter(data[:,1],data[:,2],s=100)


def plot_contour(target_function, plot, range_=(-1,1)):
    size = 50
    xx, yy = np.linspace(range_[0],range_[1],size), np.linspace(range_[0],range_[1],size) # IMPORTANT, xx, yy === x1, x2
    XX, YY = np.meshgrid(xx, yy) # IMPORTANT, XX = [[-2...20]..], YY = [[-5...-5]...[5...5]]
    ZZ = np.zeros(XX.shape)
    # IMPORTANT np.stack((XX, YY),axis=2) creates 50x50 [x1,x2]
    # print(np.stack((XX, YY),axis=2).reshape(25,2))
    # print(f"XX{XX}YY{YY}")
    ZZ = target_function(np.stack((np.ones(XX.shape),XX, YY),axis=2).reshape(size*size,3)).reshape(XX.shape)
    # plot.gca().set_aspect(1)
    plot.set_xlabel("x")
    plot.set_ylabel("y")
    plot.contourf(XX, YY, ZZ, [-1,0,1]) # the last parameter is the range of the 2 level, from -1 to 0 and from 0 to 1

#%%

class kMeanClustering:

    class Cluster:
    
        def __init__(self):
            self.pos = [np.random.uniform(-1,1,2)]
            self.points = []
        
    def __init__(self, k):
        self.k = k
        self.clusters = [self.Cluster() for _ in range(int(k))]
        self.restart = False

    def clear_points(self):
        for cluster in self.clusters: cluster.points = []

    def optimize_cluster(self):
        self.clear_points()
        for x in self.data[:,:3]:
            distance = []
            # print(x)
            # exit()
            for cluster in self.clusters:
                dist = np.linalg.norm(x[1:]-cluster.pos)
                distance.append([dist,cluster])
            min(distance,key= lambda tup: tup[0])[1].points.append(x)
            # IMPORTANT min(b)[a] where b[1] is compared if b[0] are the same

    def optimize_mean(self):
        changed = False
        for cluster in self.clusters:
            if cluster.points != []: 
                new_mean = [np.mean(np.array(cluster.points)[:,1]),np.mean(np.array(cluster.points)[:,2])]
                if cluster.pos != new_mean: changed = True
                # IMPORTANT if comparing list==list, must not use nparraylist == list
                cluster.pos = new_mean
            else: self.restart = True
        return changed

    def get_clusters_pos(self):
        pos = []
        for num in range(len(self.clusters)):
            # print(k_mean_clustering.clusters[num].pos)
            pos.append([0,self.clusters[num].pos[0],self.clusters[num].pos[1],0])
        return np.array(pos)

    def main(self,data):
        self.data = data
        self.optimize_cluster()
        while self.optimize_mean():
            if self.restart: self.restart = False; self.__init__(self.k)
            self.optimize_cluster()
        # result = [cluster.pos for cluster in self.clusters]
        # return result

class RegularRBF:

    def phi_matrix(self, data):

        gaussian_kernel = lambda X, y : np.exp(-self.gamma * np.sqrt((X[:,1] - y[1])**2 + (X[:,2] - y[2])**2))
        phi_matrix = []
        for cluster_pos in self.clusters_pos:
            # IMPORTANT np.linagle.norm return a 1d value
            phi_matrix.append(gaussian_kernel(data, cluster_pos))
            # print(data)
            # print(cluster_pos)
            # print(phi_matrix)
            # exit()
        phi_matrix = np.column_stack((np.array(phi_matrix).T,np.ones(data.shape[0])))
        # print(np.shape(phi_matrix))
        return phi_matrix

    def __init__(self, k, gamma):
        self.k = k
        self.gamma = gamma
        self.weight = None

    def fit(self, data):
        k_mean_clustering = kMeanClustering(self.k)
        k_mean_clustering.main(data)
        self.clusters_pos = k_mean_clustering.get_clusters_pos()
        # IMPORATNT, initiate np.array as zeros using X shape to ensure using nparray
        phi_matrix = self.phi_matrix(data)
        self.weight = np.linalg.pinv(phi_matrix.T @ phi_matrix) @ phi_matrix.T @ data[:,3]
        
    def predict(self, data):
        return np.sign(self.phi_matrix(data) @ self.weight)

        
#%%




#%%
data_set = DataSet()
gaussian_svm = svm.SVC(C=1e10, gamma=1.5)

count = 0
soft_margin = 0
e_in_svm, e_in_rbf, e_out_svm, e_out_rbf, svm_win, rbf_win = 0,0,0,0,0,0
k = 9
gamma = 1.5
regular_rbf = RegularRBF(k, gamma)

while count < 10:
    data_set.generate_points(100)

    regular_rbf.fit(data_set.Data)
    gaussian_svm.fit(data_set.Data[:,:3],data_set.Data[:,3])

    prediction_a = gaussian_svm.predict(data_set.X)
    e_in_svm += np.mean(data_set.Y != prediction_a)
    prediction_b = regular_rbf.predict(data_set.X)
    e_in_rbf += np.mean(data_set.Y != prediction_b)

    if np.mean(data_set.Y != prediction_b) != 0: soft_margin += 1

    data_set.generate_points(100)
    prediction_a = gaussian_svm.predict(data_set.X)
    e_out_svm += np.mean(data_set.Y != prediction_a)
    prediction_b = regular_rbf.predict(data_set.X)
    e_out_rbf += np.mean(data_set.Y != prediction_b)

    if np.mean(data_set.Y != prediction_a) < np.mean(data_set.Y != prediction_b): svm_win += 1
    else: rbf_win += 1
    # print(np.sign(e_out_rbf - e_out_svm))
    count += 1

print(f"e in is {round(e_in_svm/100,3)}")
print(f"e out is {round(e_out_svm/100,3)}")
print(f"e in with rbf is {round(e_in_rbf/100,3)}")
print(f"e out with rbf is {round(e_out_rbf/100,3)}")
print(f"percent of soft margin is {round(soft_margin/100,3)}")
print(f"svm_win is {svm_win/100}")
print(f"rbf_win is {rbf_win/100}")

fig, axes = plt.subplots(1,3,sharey=True)

for n in range(3):
    if n == 0:plot_contour(data_set.target_function, axes[n], (-1,1))
    if n == 1:plot_contour(regular_rbf.predict, axes[n], (-1,1))
    if n == 2:plot_contour(gaussian_svm.predict, axes[n], (-1,1))
    plot_points(regular_rbf.clusters_pos, axes[n])
    plot_points(data_set.Data, axes[n])

plt.show()
