import requests
import numpy as np
import random
# IMPORTANT import sklearn as sk doesn't import all package from sk
from sklearn import svm, model_selection
import cvxopt
import seaborn as sb
import matplotlib.pyplot as plt
#%%
# try:
#     with open("features.train", "x") as f_in:
#         request_in = requests.get("http://www.amlbook.com/data/zip/features.train")
#         f_in.write(request_in.text)
# except FileExistsError as e:
#     print("Training data already downloaded")

# try:
#     with open("features.test","x") as f_out:
#         request_out = requests.get("http://www.amlbook.com/data/zip/features.test")
#         f_out.write(request_out.text)
# except FileExistsError as e:
#     print("Test data already downloaded")

#%%
# IMPORTANT file.read returns a string of all items, file.readlines return a list of lines(strings)
# train_data_ = list(map(float, train_data.read()[:].strip().split())
# print(train_data_)

# IMPORTANT::
# with open("features.train") as train_data:
    # data_train_ = np.array([list(map(float,line.strip().split())) for line in train_data])
def sb_scatter():
    data_train = np.loadtxt("features.train")
    sb.scatterplot(data_train[:,1], data_train[:,2], data_train[:,0])
    plt.show()
# sb_scatter()
#%%
def subplots_():
    data_train = np.loadtxt("features.train")
    get_pos = lambda n: divmod(n, 3)
    fig, ax = plt.subplots(4, 3, figsize=(13,13), sharex=True, sharey=True)

    for n in range(10):
        i = np.where(data_train[:,0] == n)
        ax[get_pos(n)].scatter(data_train[i,1], data_train[i,2], s=1)
        ax[get_pos(n)].set_title(n)
        ax[get_pos(n)].set_xlabel("intensity")
        ax[get_pos(n)].set_ylabel("symmetry")

    plt.show()
# subplots_()
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
# polynomial_svc = svm.SVC(C=0.01,degree=2, kernel='poly',gamma=1)

# for a in range(9):
#     data_set = one_vs_all(a)
#     polynomial_svc.fit(data_set[:,1:],data_set[:,0])
#     prediction = polynomial_svc.predict(data_set[:,1:])
#     if a == 2: sv_num_2 = len(polynomial_svc.support_)
#     if a == 1: sv_num_1 = len(polynomial_svc.support_)
#     E_in = np.mean(data_set[:,0] != prediction)
#     print(f"num is {a} error is {E_in}")
    
# print(f"difference in sv num is {sv_num_2-sv_num_1} ")
# IMPORTANT svc.fit resets weight, use partial_fit() to preserve previous stuff

#%%
# c_list = (1e0,1e-1,1e-2,1e-3)
c_list = (0.01,1,100,1e4,1,1e6)
# IMPORTANT, 10e0 == 10 != 1 
# for a ,b in np.ndindex(9,9): # IMPORTANT, this is the concise way of using two for loops

for c_val in c_list:
    # for q_val in (2,5):
    polynomial_svc = svm.SVC(C=c_val, kernel='rbf',gamma=1)
    data_train = np.loadtxt("features.train")
    data_set = one_vs_one(1,5,data_train)
    polynomial_svc.fit(data_set[:,1:],data_set[:,0])
    prediction = polynomial_svc.predict(data_set[:,1:])
    E_in = np.mean(data_set[:,0] != prediction)
    
    data_test_ = np.loadtxt("features.test")
    data_test_ = one_vs_one(1,5,data_test_)
    prediction = polynomial_svc.predict(data_test_[:,1:])
    E_out = np.mean(data_test_[:,0] != prediction)
    
    print(f"c val is {c_val} \nerror in is {E_in} error out is {E_out}")
    
# print(f"difference in sv num is {sv_num_2-sv_num_1} ")

# %%
# c_list = (0.01,1,100,1e4,1,1e6)
# data_train = np.loadtxt("features.train") # ISSUE: one extra data?
# data_train = one_vs_one(1,5,data_train)
# E_vali_list = []
# for c_val in c_list:
#     polynomial_svc = svm.SVC(C=c_val, kernel='rbf',gamma=1)
#     split_pos = model_selection.KFold(n_splits=10, shuffle=True)
    
#     E_vali = 0
#     for data_train_, data_vali in split_pos.split(data_train):
#         polynomial_svc.fit(data_train[data_train_,1:],data_train[data_train_,0])
#         prediction = polynomial_svc.predict(data_train[data_vali,1:])
#         E_vali += np.mean(data_train[data_vali,0] != prediction)
#     E_vali /= split_pos.n_splits
#     E_vali_list.append((c_val,E_vali))
# print(f"list is {E_vali_list} \nminimum is {min(np.array(E_vali_list)[:,1])}")
    

        



