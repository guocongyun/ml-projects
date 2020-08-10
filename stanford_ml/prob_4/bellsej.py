import sounddevice as sd
import numpy as np
import pickle

Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('./stanford_ml/prob_4/mix.dat')
    return mix

def play(vec):
    sd.play(vec, Fs, blocking=True)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    # IMPORTANT slowly decreasing learning rate to speed up learning
    print('Separating tracks ...')
    ######## Your code here ##########

    for learning_rate in anneal:
        print(f"staring epoch with learning rate {learning_rate}")
        for row in X:
            row = row.reshape(5,1)
            # IMPORTANT matrices in numpy are row matrices, default 1,5 as aooise to column matrices in stanford 5,1
            assert row.shape == (5,1), row.shape
            assert W.shape == (5,5),W.shape
            sigmoid_deriv = np.array([1 - 2*sigmoid(weight.reshape(1,5).flatten() @ row) for weight in W]) @ row.T; assert sigmoid_deriv.shape == (5,5),sigmoid_deriv.shape
            WT_inv = np.linalg.inv(W.T); assert WT_inv.shape == (5,5),W.shape
            W += learning_rate * (sigmoid_deriv + WT_inv)

    with open('./stanford_ml/prob_4/ICA_weights','w+b') as weight_file: pickle.dump(W, weight_file)

    ###################################
    return W

def unmix(X, W):
    S = np.zeros(X.shape)

    ######### Your code here ##########
    S = X @ W
    print(W)
    ##################################
    return S

def main():
    X = normalize(load_data())

    # for i in range(1):
    #     print('Playing mixed track %d' % i)
    #     play(X[:, i])

    # p = np.random.permutation(X.shape[0])
    # X_shuffle = X[p,:]
    # W = unmixer(X)
    W_expected = np.array([[ 72.15081922,  28.62441682,  25.91040458, -17.2322227 , -21.191357  ],
                       [ 13.45886116,  31.94398247,  -4.03003982, -24.0095722 , 11.89906179 ],
                       [ 18.89688784,  -7.80435173,  28.71469558,  18.14356811, -21.17474522],
                       [ -6.0119837 ,  -4.15743607,  -1.01692289,  13.87321073, -5.26252289 ],
                       [ -8.74061186,  22.55821897,   9.61289023,  14.73637074, 45.28841827 ]])
    with open('./stanford_ml/prob_4/ICA_weights','rb') as weight_file: W = pickle.load(weight_file)
    assert ((W - W_expected) < 1e-8).all(), W
    S = normalize(unmix(X, W))

    for i in range(X.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])

if __name__ == '__main__':
    main()
