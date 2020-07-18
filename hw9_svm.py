import requests
import numpy as np
import random
from sklearn import svm, model_selection, linear_model
import seaborn as sb
import matplotlib.pyplot as plt

#%%
phi = lambda dataset: np.array([
    dataset[:,1]**2 - 2*dataset[:,0] - 1,
    dataset[:,0]**2 - 2*dataset[:,1] + 1,
    dataset[:,2]
]).T 
# IMPORTANT since the transformed parameter is smaller than paramters in 2degree polynomial kernel therefore we found less support vectors

def sb_scatter():
    data_train = np.loadtxt("custom_set.train")
    data_train = phi(data_train)
    above = (data_train[:,2] > 0) 
    below = (data_train[:,2] < 0) 
    # sb.scatterplot(data_train[:,0],data_train[:,1],data_train[:,2])
    plt.scatter(data_train[above,0],data_train[above,1])
    plt.scatter(data_train[below,0],data_train[below,1])
    print(data_train)
    plt.show()

# sb_scatter()
#%%
data_train = np.loadtxt("custom_set.train") # svm internally uses libsvm
poly_svm = svm.SVC(coef0=1,kernel='poly',degree=2,gamma=1)
poly_svm.fit(data_train[:,:2],data_train[:,0])
print(len(poly_svm.support_))
