import numpy as np
import matplotlib.pyplot as plt
import pickle


def readData(images_file, labels_file):
    # IMPORTANT delimiter is the string used to seperate values
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels.T)
    return cost, accuracy

def compute_accuracy(output, labels):
    accuracy = np.sum(np.argmax(output,axis=0) == np.argmax(labels,axis=0)) * 1. / labels.shape[1] 
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

#%%

def softmax(x):
    """
    compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
	# IMPORTANT to prevent overflow, i.e. exp(790)=infinity we substract the max(x) on both side
    # IMPORTANT keepdims retains the previous struct i.e. max([0,1,2],keepdim=True) = [2] instead of 2
    # IMPORTANT if x != -10000 exp(x) != 0
    C = np.max(x,axis=0, keepdims=True)
    x -= C
    s = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
    return s

def sigmoid(x):
    """
    compute the sigmoid function for the input here.
    """
    s = 1/(1+np.exp(-x))
    return s

def cross_entropy_loss(y, y_):
    """
    compute and return the cross entropy loss
    """
    # assert y.shape == y_.shape == (10,1000), (y.shape,y_.shape)
    # IMPORTANT regularizing bias restrict it's intercept ad unecesseray because it doesn't depend on x
    log_probs= y[y_>0] * np.log(y_[y_>0])
    loss = -np.sum(log_probs)/y.shape[1]
    return loss

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1'] 
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # assert data.shape == (1000,784), data.shape
    Z1 = W1 @ data.T  + b1 # 300x1000
    h = sigmoid(Z1)
    Z2 = W2 @ h + b2
    y = softmax(Z2)
    cost = cross_entropy_loss(labels.T,y)
    return h, y, cost

def backward_prop(data, labels, params, lambd=0):
    """
    return gradient of parameters
    """

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    assert W1.shape == (300,784), W1.shape
    assert b1.shape == (300,1), b1.shape
    assert W2.shape == (10,300), W2.shape
    assert b2.shape == (10,1), b2.shape

    h, y, _ = forward_prop(data, labels, params)
    assert h.shape == (300, data.shape[0]), h.shape
    assert y.shape == (10, data.shape[0]), y.shape

    # IMPORTANT for each data softmax func output a 10 size list of 10 different outcomes hence 1000x10
    sigma2 = y - labels.T # 10x1000
    gradb2 = np.mean(sigma2, axis=1, keepdims=True) # 10x1
    gradW2 = (sigma2 @ h.T)/ data.shape[0] + 2 * lambd * W2 # 10x300

    sigma1 = W2.T @ sigma2 * (h * (1-h)) # 300x1000
    # here through matrix multiplication it has already taken the sum of sigma2j*wj
    # IMPORTANT 1 sigma per output, 1 (h * (1-h)) per input
    gradb1 = np.mean(sigma1, axis=1, keepdims=True) # 300x1
    gradW1 = (sigma1 @ data) / data.shape[0] + 2 * lambd * W1 # 300x700 
    # IMPORTANT here we have already sum the 1000 data points via 300x1000 @ 1000x700 dot product
    # IMPORTANt one xi per input
    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad

def update_params(grad,params,learning_rate):
    """
    update the parameters according to the grads
    """
    params['W1'] -= learning_rate * grad['W1']
    params['b1'] -= learning_rate * grad['b1']
    params['W2'] -= learning_rate * grad['W2']
    params['b2'] -= learning_rate * grad['b2']


def nn_train(trainData, trainLabels, devData, devLabels):
#%%
    (m, n) = trainData.shape
    epoch_num = 30
    num_hidden = 300
    learning_rate = 5
    mini_batch_size = 1000
    outputdim = 10
    iteration = m//mini_batch_size
    params = {}
    assert trainData.shape == (50000,784), trainData.shape
    assert trainLabels.shape == (50000,10), trainLabels.shape    
    params['W1'], params['b1'] = np.random.randn(num_hidden,n), np.zeros((num_hidden,1)) # IMPORTANT (1,x) instead of 1d list
    params['W2'], params['b2'] = np.random.randn(outputdim,num_hidden), np.zeros((outputdim,1))
    # IMPORTANT, the outputdim is 10

    stats = {
        'train_cost' : [],
        'train_acc' : [],
        'test_cost' : [],
        'test_acc' : []
    }
    
    for run in range(epoch_num): 
        print(f"starting epoch {run}")
        for num in range(1,iteration+1):
            mini_batch_data = trainData[(num-1)*mini_batch_size:num*mini_batch_size]
            mini_batch_label = trainLabels[(num-1)*mini_batch_size:num*mini_batch_size]
            grad = backward_prop(mini_batch_data, mini_batch_label, params, lambd=0.001)
            update_params(grad , params, learning_rate)
        train_cost, train_accuracy = nn_test(trainData, trainLabels, params)
        test_cost, test_accuracy = nn_test(devData, devLabels, params)
        stats['train_cost'].append(train_cost)
        stats['train_acc'].append(train_accuracy)
        stats['test_cost'].append(test_cost)
        stats['test_acc'].append(test_accuracy)

        with open("../data/project_4/parameter","w+b") as parameter: pickle.dump(params, parameter)
        # IMPORTANT add + in open mode to allow creating file, add b in open mode to open in binary mode for pickle
        with open("../data/project_4/stats","w+b") as stat: pickle.dump(stats, stat)

            # for key, value in params.items(): params_[key] = value.tolist(); parameter.write(str(params_))
#%%
    with open("../data/project_4/stats","rb") as stat: stats = pickle.load(stat)
    with open("../data/project_4/stats_reg","rb") as stat: stats_reg = pickle.load(stat)
    with open("../data/project_4/stats_reg_2","rb") as stat: stats_reg_2 = pickle.load(stat)
#%%
    fig, axes = plt.subplots(2,3) # IMPORTANT this returns 2d array
    axes[0,0].plot(range(1,31), stats['train_cost'], label='training cost')
    axes[0,0].plot(range(1,31), stats['test_cost'], label='testing cost')
    axes[0,1].plot(range(1,31), stats_reg['train_cost'], label='training cost reg')
    axes[0,1].plot(range(1,31), stats_reg['test_cost'], label='testing cost reg')
    axes[0,2].plot(range(1,31), stats_reg_2['train_cost'], label='training cost reg')
    axes[0,2].plot(range(1,31), stats_reg_2['test_cost'], label='testing cost reg')
    axes[1,0].plot(range(1,31), stats['train_acc'], label='training acc')
    axes[1,0].plot(range(1,31), stats['test_acc'], label='testing acc')
    axes[1,1].plot(range(1,31), stats_reg['train_acc'], label='training acc reg')
    axes[1,1].plot(range(1,31), stats_reg['test_acc'], label='testing acc reg')
    axes[1,2].plot(range(1,31), stats_reg_2['train_acc'], label='training acc reg')
    axes[1,2].plot(range(1,31), stats_reg_2['test_acc'], label='testing acc reg')
    for axe in axes: 
        for ax in axe: ax.legend()
    plt.show()
    return params

def main():
#%%
    # np.random.seed(100)
    # trainData, trainLabels = readData('../data/project_4/images_train.csv', '../data/project_4/labels_train.csv')
    # trainLabels = one_hot_labels(trainLabels)
    # p = np.random.permutation(60000)
    # trainData = trainData[p,:]
    # trainLabels = trainLabels[p,:]

    # # IMPORTANT this makes the data random shuffled everytime read

    # devData = trainData[0:10000,:]
    # devLabels = trainLabels[0:10000,:]
    # trainData = trainData[10000:,:]
    # trainLabels = trainLabels[10000:,:]

    # mean = np.mean(trainData)
    # std = np.std(trainData)
    # trainData = (trainData - mean) / std
    # devData = (devData - mean) / std

    testData, testLabels = readData('../data/project_4/images_test.csv', '../data/project_4/labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - np.mean(testData)) / np.std(testData)
# %%
    # trainData = 0
    # trainLabels = 0
    # devData = 0
    # devLabels = 0
#%%
    # params = nn_train(trainData, trainLabels, devData, devLabels)

    readyForTesting = True
    if readyForTesting:
        # with open("../data/project_4/parameter","rb") as parameter: params = pickle.load(parameter)
        with open("../data/project_4/parameter_reg","rb") as parameter: params_reg = pickle.load(parameter)
        # with open("../data/project_4/parameter_reg_2","rb") as parameter: params_reg_2 = pickle.load(parameter)
        # cost, accuracy = nn_test(testData, testLabels, params)
        # print('Test accuracy: %f' %accuracy)
        cost, accuracy = nn_test(testData, testLabels, params_reg)
        print('Test accuracy with reg: %f' %accuracy)
        # cost, accuracy = nn_test(testData, testLabels, params_reg_2)
        # print('Test accuracy with lots of reg: %f' %accuracy)

if __name__ == '__main__':
    main()


# %%
