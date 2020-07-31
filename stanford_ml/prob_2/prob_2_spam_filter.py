from __future__ import division
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

# trainMatrix, tokenlist, trainCategory = readMatrix('./stanford_ml/prob_2/MATRIX.TRAIN')
testMatrix, tokenlist, testCategory = readMatrix('./stanford_ml/prob_2/MATRIX.TEST')

# print(trainMatrix.shape)
# print(trainCategory) # assume 1 is spam

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print ('Error: %1.4f' % error)
    return error

def evaluate_(matrix, test_matrix, category):
    # state = {}
    ######

    laplace_smoothing = 1
    spam = matrix[(category==1)]
    ham = matrix[(category==0)]

    spam_len = sum(category)
    ham_len = len(category) - sum(category)

    px_given_spam = lambda tokenlist: np.exp(sum([np.log((np.sum(spam[:,index],axis=0) + laplace_smoothing)/(spam_len + 2*laplace_smoothing)) for index, value in enumerate(tokenlist) if value == 1]))
    px_given_ham = lambda tokenlist: np.exp(sum([np.log((np.sum(ham[:,index],axis=0) + laplace_smoothing)/(ham_len + 2*laplace_smoothing)) for index, value in enumerate(tokenlist) if value == 1]))
    # IMPORATNT after calculating using log, must use exp again to get the prob back
    pspam = (spam_len + laplace_smoothing) / (len(category) + 2*laplace_smoothing)

    pspam_given_x = lambda tokenlist: px_given_spam(tokenlist) * pspam / (px_given_spam(tokenlist) * pspam + px_given_ham(tokenlist) * (1-pspam))
    pham_given_x = lambda tokenlist: px_given_ham(tokenlist) * (1-pspam) / (px_given_spam(tokenlist) * pspam + px_given_ham(tokenlist) * (1-pspam))

    output = np.zeros(test_matrix.shape[0])
    for index, row in enumerate(test_matrix):
        if (pspam_given_x(row) > 0.5):
            output[index] = 1

    # state['px_given_spam'] = np.log((np.sum(spam,axis=0) + laplace_smoothing)/(spam_len + 2*laplace_smoothing))
    # state['px_given_ham'] = np.log((np.sum(ham,axis=0) + laplace_smoothing)/(ham_len + 2*laplace_smoothing))
    # state['pspam'] = (spam_len + laplace_smoothing) / (len(category) + 2*laplace_smoothing)
    # spam_token = np.argsort(state['px_given_spam'] - state['px_given_ham'])[-5:]
    # print(spam_token)
    # print(np.array(tokenlist)[spam_token])

    return output


train_sizes = np.array([50, 100, 200, 400, 800, 1400])
errors = np.ones(train_sizes.shape)
for i,train_size in enumerate(train_sizes):
    # trainMatrix, tokenlist, trainCategory = readMatrix('./stanford_ml/prob_2/MATRIX.TRAIN')
    trainMatrix, tokenlist, trainCategory = readMatrix('./stanford_ml/prob_2/MATRIX.TRAIN.'+str(train_size))
    output = evaluate_(trainMatrix, testMatrix, trainCategory)
    evaluate(output, testCategory)

    svm_ = svm.SVC(kernel='rbf',C=1e100, gamma=1e-10)
    svm_.fit(trainMatrix, trainCategory)
    evaluate(svm_.predict(testMatrix), testCategory)