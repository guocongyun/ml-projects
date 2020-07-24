import requests
import numpy as np
import random
from sklearn import svm, model_selection, linear_model
import cvxopt
import seaborn as sb
import matplotlib.pyplot as plt
#%%
def one_vs_one(digit_a,digit_b, dataset):
    data_a = dataset[(dataset[:,0] == digit_a)]
    data_b = dataset[(dataset[:,0] == digit_b)]
    data_a[:,0] = 1
    data_b[:,0] = -1
    data_set = np.vstack((data_a,data_b))
    return data_set
# one_vs_one(1,3)
#%%
def one_vs_all(digit_a, dataset):
    data_a = dataset[(dataset[:,0] == digit_a)]
    data_b = dataset[(dataset[:,0] != digit_a)]
    data_a[:,0] = 1
    data_b[:,0] = -1
    data_set = np.vstack((data_a,data_b))
    return data_set

# one_vs_all(1)

#%%
# data_train = np.loadtxt("features.train")
# data_test = np.loadtxt("features.test")
# clf = linear_model.RidgeClassifier(alpha=1)

def train_test (train_data_set, test_data_set, transform = None):
    if transform != None: train_data_set = transform(train_data_set)
    clf.fit(train_data_set[:,1:],train_data_set[:,0])
    prediction = clf.predict(train_data_set[:,1:])
    e_in = np.mean(train_data_set[:,0] != prediction)

    if transform != None: test_data_set = transform(test_data_set)
    prediction = clf.predict(test_data_set[:,1:])
    e_out = np.mean(test_data_set[:,0] != prediction)

    return e_in, e_out

phi = lambda data_set: np.array([
    data_set[:,0],
    np.ones(data_set.shape[0]),
    data_set[:,1],
    data_set[:,2],
    data_set[:,1]*data_set[:,2],
    data_set[:,1]**2,
    data_set[:,2]**2]).T

for alpha in [1,0.01]:
    # IMPORTANT need to transpose the data to have the original shape
    clf = linear_model.RidgeClassifier(alpha=alpha)
    train_data_set = one_vs_one(1,5, data_train)
    test_data_set = one_vs_one(1,5, data_test)
    e_in, e_out = train_test(train_data_set, test_data_set)
    print(f"alpha is {alpha} e in is {round(e_in,3)}")
    print(f"alpha is {alpha} e out is {round(e_out,3)}")
    e_in_phi, e_out_phi = train_test(train_data_set, test_data_set, phi)
    print(f"num is {num} e in with trans is {round(e_in_phi,3)}")
    print(f"num is {num} e out with trans is {round(e_out_phi,3)}")
    print(f"improvement of e out is {round(e_out_phi/e_out,3)}")

    
