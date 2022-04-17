import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

n_inputs = 2
n_neurons = 4

weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))

print(weights)
print(biases)

X, y = spiral_data(samples = 100, classes = 3)

dense1 = layer_dense(2, 3)
dense1.forward(X)
print(dense1.output[:5])