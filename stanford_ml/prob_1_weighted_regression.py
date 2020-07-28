import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('./stanford_ml/quasar_train.csv')
# IMPOTRANT python interactive root is at stanford_ml whereas terminal root is at MachinelearningHw
# print(cols_train)
#%%
weight = np.zeros(3)
y = df_train.loc[:].to_numpy().T # head() returns a list of list, loc returns a list

X = df_train.columns.values.astype(float)
X = np.vstack((np.ones(X.shape[0]), X.T)).T # adding the constant term

#%%
def fit(X, y):
    # print(X.shape)
    # print(y.shape)
    pesudo_inverse  = (np.linalg.inv(X.T @ X)) @ X.T
    return pesudo_inverse @ y

#%%

def fit_weighted(X, y, tau=5): # NOTE smaller tau, no less than 0.9 means more overfitting
    prediction = []
    for _, xi in enumerate(X):
        weight_matrix = np.diag(np.exp(-(X[:,1] - xi[1])**2 / (2*tau**2)))
        # IMPORATANT np.diag turns a n length list into a nxn matrix
        # IMPORTANT weighted regression have different weight and ceta for different xi
        pesudo_inverse  = (np.linalg.inv(X.T @ weight_matrix @ X)) @ X.T @ weight_matrix
        prediction.append(xi @ pesudo_inverse @ y)
    # print(np.exp(-(X[:,1] - xi[1])**2 / (2*tau**2)))
    # print(np.diag(np.exp(-(X[:,1] - xi[1])**2 / (2*tau**2))))
    return np.array(prediction)


# print(weight.shape)
# print(y.shape)
# print(weight[:,0])
#%%

prediction = fit_weighted(X, y, 5) # NOTE reshape can not change how value in list are stored, i.e. [[1,2],[3,4]] can't be [[3,4],[4,2]]
df_train = pd.DataFrame(prediction, index=X[:,1])
wave_length_right = X[:,1] >= 1300
wave_length_left = X[:,1] < 1200
df_train_right = df_train[wave_length_right].T
df_train_left = df_train[wave_length_left].T

df_test = pd.read_csv('./stanford_ml/quasar_test.csv')
X = df_test.columns.values.astype(float)
X = np.vstack((np.ones(X.shape[0]), X.T)).T # adding the constant term
y = df_test.loc[:].to_numpy().T # head() returns a list of list, loc returns a list
prediction = fit_weighted(X, y, 5) # NOTE reshape can not change how value in list are stored, i.e. [[1,2],[3,4]] can't be [[3,4],[4,2]]
df_test = pd.DataFrame(prediction, index=X[:,1])

df_test_right = df_test[wave_length_right].T
df_test_left = df_test[wave_length_left].T
# VERY IMPORTANT, if index of column name of two dataframe is different in type or value, calculation when result in Nan
def ker(t):
    return np.max(1-t, 0)

def distance(f1, f2):
    # print(f1)
    # print(f2)
    # print(f1.shape)
    # print(f2.shape)
    # print((f1 - f2)**2)
    return np.sum((f1 - f2)**2,axis=1)
#%%
err = []
predictions = []
for index, row in df_test_right.iterrows():
    dist = distance(df_train_right, row)
    # IMPORTANT, sum over axis=1 is sum vertically axis = 0 sums horizontally 
    neighb = dist.sort_values()[:3]
    # sum by default have axis=None which will sum all of the element in input array
    max_d = dist.max()
    eq1 = np.sum([ker(item/max_d) * df_train_left.loc[k] for (k, item) in neighb.iteritems()], axis=0)
    # IMPORTANT iteritems or items only works in dict or dataframe, which gaves the index (not 1,2,3) and the item 
    # enumerate works on list which gives the 1,2,3, index and the value
    eq2 = np.sum([ker(item/max_d) for (k, item) in neighb.iteritems()], axis=0)

    f_left_hat = eq1/eq2
    predictions.append(f_left_hat)
    err.append(np.sum((f_left_hat - df_test_left.loc[index]) ** 2))

print(np.mean(err))

#%%
# plt.plot(X[:, 1], y[:,0])
# plt.plot(X[:, 1], prediction[:,0])
# IMPORTANT, plt.plot first paramter is the x-axis

#%%
fig, axes = plt.subplots(2,2)
axes = axes.ravel()

for num in range(4):
    # print(prediction)
    # print(prediction.shape)
    # exit()
    axes[num].plot(X[wave_length_left,1], df_test_left.loc[num], label="actual value")
    axes[num].plot(X[wave_length_left,1], predictions[num], label="predicted")
    axes[num].legend()
plt.show()

