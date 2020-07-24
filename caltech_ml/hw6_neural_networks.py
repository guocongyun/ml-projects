import numpy as np
from graphviz import Digraph, Source
import matplotlib.pyplot as plt
from IPython.display import display
from sympy import diff, symbols

class Layer:

    def __init__(self, neurons, prev_layer_size = 0):
        self.size = neurons + 1
        self.weights = np.array([np.zeros(prev_layer_size) for _ in range(self.size)])
        self.output = np.ones(self.size)
        self.delta = np.ones(self.size)

class NeuralNetwork:
    
    def __init__(self):
        self.layers = []
        self.prediction = 0.0
        self.actual_value = 0.0
        self.learning_rate =0.1
        self.weight_decay_rate = 0.0001

    def input_layer(self, neurons):
        layer = Layer(neurons)
        layer.output[0] = 1
        self.layers.append(layer)

    def add_layer(self, neurons):
        layer = Layer(neurons, self.layers[-1].size)
        layer.weights[0] = 1
        layer.output[0] = 1
        self.layers.append(layer)
        layer.weights[1:] = self.xavier_initialization()

    def output_layer(self, neurons=1):
        layer = Layer(neurons-1, self.layers[-1].size)
        prev_layer_size = self.layers[len(self.layers)-1].size
        # print("size is {}".format(layer.size))
        layer.weights[0:] = np.random.randn(neurons, prev_layer_size) * np.sqrt(1/(prev_layer_size))
        self.layers.append(layer)

            
    def xavier_initialization(self):
        layer_index = len(self.layers) - 1
        last_layer_size = self.layers[layer_index].size - 1
        prev_layer_size = self.layers[layer_index-1].size
        weights = np.random.randn(last_layer_size, prev_layer_size) * np.sqrt(1/(prev_layer_size))
        # randn(3,2) generated  3*2 matrix of random float from distribution N(0,1)
        return weights

    def theta(self, s):
        return np.tanh(s) 
        # IMPORTNAT np.tanh only support floats, not symbols

    def theta_deriv(self, s):
        return 1-np.tanh(s)**2

    def err_func(self, x, y):
        return (x - y)**2   

    def err_deriv(self, x, y):
        u, v = symbols("u v")
        derivative = lambda x, y: np.array(diff(self.err_func(u,v),u).subs({u:x, v:y}))
        return derivative(x, y)

    def predict(self, x):
        assert len(x) == self.layers[0].size - 1
        for index in range(len((self.layers))):
            layer = self.layers[index]
            if layer == self.layers[0]:
                layer.output[1:] = x
            else:
                input_ = self.layers[index-1].output

                signal = layer.weights @ np.transpose(input_)
                output = self.theta(signal)
                layer.output = output

                if index == len(self.layers)-1:
                    layer.output[0] = output
                    self.prediction = layer.output[0]

    def graphviz(self): # ISSUE how to rearrange the nodes
        dot = Digraph(engine="neato")
        L = len(self.layers) - 1
        for layer in range(0, L+1):
            # Nodes
            for node in range(self.layers[layer].size):
                dot.node("x({})[{}]".format(layer, node), pos="{},{}!".format(node,-1*layer))
                # Forward edges
                if layer < L:
                    for i in range(0, self.layers[layer+1].size):
                        w = self.layers[layer+1].weights[i][node]
                        # IMPORTANT, for np arrays only list[a,b] === list[a][b]
                        if w < 0:
                            color = "red"
                            w = -w
                        else:
                            color = "blue"
                        dot.edge(
                            "x({})[{}]".format(layer, node), "x({})[{}]".format(layer+1, i),
                            arrowhead="none",
                            penwidth="{}".format(w),
                            color=color
                        )
        # display(dot) 
        s = Source(dot,filename="neural_network.gv",format="png")
        s.view()

    def backpropagation(self, y):
        self.actual_value = y

        L = len(self.layers) - 1
        layer = self.layers[L] 
        prev_layer = self.layers[L - 1] 

        err_deriv = self.err_deriv(self.prediction,self.actual_value)
        signal = layer.weights[0] @ np.transpose(prev_layer.output)
        theta_deriv = self.theta_deriv(self.theta(signal))
        self.layers[L].delta[0] = err_deriv * theta_deriv


        for index in range(L-1,0,-1): # doesn't count the input layer
            layer = self.layers[index] 
            prev_layer = self.layers[index - 1] 
            next_layer = self.layers[index +  1]

            input_ = prev_layer.output
            for this_ in range(layer.size):
                self.layers[index].delta[this_] = 0
                for next_ in range(next_layer.size):
                    # print("what")
                    # print(next_)
                    # print(next_layer.weights[next_][this_])
                    # print(next_layer.delta)
                    # print(next_layer.delta[0])
                    # print((1 - (input_)**2) * next_layer.weights[next_][this_] * next_layer.delta[next_])
                    # print(self.layers[index].delta[this_])
                    self.layers[index].delta[this_] = (1 - (input_[this_])**2) * next_layer.weights[next_][this_] * next_layer.delta[next_]


            # IMPORTANT numpy functions require float type argument explicity


    def update(self):
        for index in range(1,len(self.layers)): # doesn't count the input layer
            layer = self.layers[index] # IMPORTANT, list are passed by reference in python not value
            prev_layer = self.layers[index - 1]
            for num in range(len(layer.weights)): # IMPORTANT only use _ in for loop if _ meant to be discarded
                # print(layer.weights[_])
                weight = layer.weights[num]
                input_ = prev_layer.output
                weight_decay = - 2 * self.learning_rate * weight * self.weight_decay_rate
                layer.weights[num] = weight - self.learning_rate * input_ * layer.delta[num] - weight_decay


    def train(self, rand_mean):
        total_error = 0
        iter_num = 0
        for _ in range(TRAIN):
            # rand_mean = np.random.uniform(-1,1)
            # rand_var = np.random.uniform(0,1)
            rand_sample = np.random.normal(rand_mean,1,10)
            
            self.predict(rand_sample)
            self.backpropagation(rand_mean)
            iteration_error = self.err_func(self.prediction,self.actual_value)
            total_error += iteration_error
            iter_num += 1
            self.update()
            # if iteration_error < 0.00000000001:
            #     break
        print(iter_num)
        print(rand_mean)
        print(self.prediction)
        print(self.err_func(self.prediction,self.actual_value))
        return total_error/TRAIN

    def test(self, rand_mean):
        total_error = 0
        print("rand mean is: {}".format(rand_mean))
        for _ in range(TEST):

            rand_sample = np.random.normal(rand_mean,1,10)
            self.predict(rand_sample)
            if _%100 == 0: print(self.prediction)
            iteration_error = self.err_func(self.prediction,self.actual_value)
            total_error += iteration_error

        return total_error/TEST

    def main(self):
        total_insample_error = 0
        total_outsample_error = 0
        for _ in range(RUNS):
            rand_mean = np.random.uniform(-1,1)
            total_insample_error += self.train(rand_mean)
            total_outsample_error += self.test(rand_mean)
        total_insample_error /= RUNS
        total_outsample_error /= RUNS
        print(self.layers[1].weights) # IMPORTANT, neural network need to update the bias term i.e. the x0=1 weights/ bias
        print("total in sample error is: {}".format(total_insample_error))
        print("total out sample error is: {}".format(total_outsample_error))

TRAIN = 3000
TEST = 3000
RUNS = 1

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    neural_network.input_layer(10)
    neural_network.add_layer(2)
    neural_network.output_layer(1)
    neural_network.main()
    neural_network.graphviz()
    

    
    

