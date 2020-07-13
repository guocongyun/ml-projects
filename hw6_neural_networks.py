import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt

class Layer:

    def __init__(self, neurons):
        self.size = neurons + 1
        self.weights = np.zeros(self.size)
        self.input = np.zeros(self.size)
        self.output = np.zeros(self.size)

class NeuralNetwork:
    
    def __init__(self):
        self.layers = []

    def input_layer(self, neurons):
        pass

    def add_layer(self, neurons):
        layer = Layer(neurons)
        layer.weights[0] = 1
        layer.weights[0:] = self.xavier_initialization(layer.size)
            
    def xavier_initialization(self, size):
        weights = np.random.randn(size, size-1) * np.sqrt(1/(size-1))
        # randn(3,2) generated  3*2 matrix of random float from distribution N(0,1)
        print(weights)
        return weights

    def theta(self, x):
        return np.tanh(x)
    
    def predict(self):
        pass

    def backpropagation(self):
        pass

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    neural_network.add_layer(2)

    
    

