import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt
from sympy import diff, symbols

class Layer:

    def __init__(self, neurons, prev_layer_size = 0):
        self.size = neurons + 1
        self.weights = np.array([np.zeros(prev_layer_size) for _ in range(self.size)])
        self.output = np.ones(self.size)
        self.delta = 0

class NeuralNetwork:
    
    def __init__(self):
        self.layers = []
        self.prediction = 0.0
        self.actual_value = 0.0
        self.learning_grade =0.1

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
        print(type(s))
        return np.tanh(s) # IMPORTNAT np.tanh only support floats

    def theta_deriv(self, s):
        u = symbols("u")
        derivative = lambda s: np.array(diff(self.theta(u),u).subs({u:s}))
        return derivative(s)

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
                for neuron in range(layer.size):
                    signal = layer.weights[neuron] @ np.transpose(input_)
                    output = self.theta(signal)
                    layer.output[0:] = output
                if index == len(self.layers)-1:
                    print("layer output is: %s"%layer.output)
                    self.prediction = layer.output



    def backpropagation(self, y):
        self.actual_value = y
        for index in range(1,len(self.layers)): # doesn't count the input layer
            layer = self.layers[len(self.layers) - index] # backwards
            prev_layer = self.layers[len(self.layers) - index - 1] # backwards
            if layer == self.layers[-1]:
                print(self.prediction)
                print(self.actual_value)
                err_deriv = self.err_deriv(self.prediction,self.actual_value)
                signal = layer.weights[0] @ np.transpose(prev_layer.output)
                print(signal)
                theta_deriv = self.theta_deriv(signal)
                layer.delta = err_deriv * theta_deriv
                # IMPORTANT numpy functions require float type argument explicity

            else:
                input_ = prev_layer.output
                for neuron in range(layer.size):
                    layer.delta += (1 - (input_)**2) * layer.weights[neuron] * self.layers[len(self.layers) - index + 1 ].delta

    def update(self):
        for index in range(1,len(self.layers)): # doesn't count the input layer
            layer = self.layers[len(self.layers) - index]
            prev_layer = self.layers[len(self.layers) - index - 1]
            for _ in range(len(layer.weights)):
                weight = layer.weights[_]
                input_ = prev_layer.output
                weight = weight - self.learning_grade * input_ * weight

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    neural_network.input_layer(3)
    neural_network.add_layer(2)
    neural_network.output_layer(1)
    neural_network.predict([2,2,2])
    neural_network.backpropagation(1)
    neural_network.update()
    

    
    

