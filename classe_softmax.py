import numpy as np
from nnfs.datasets import spiral_data
import nnfs

class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(singe_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

if __name__ == '__main__':
    softmax = Activation_Softmax()
    softmax.forward([[1, 2, 3]])
    print(softmax.output)
    softmax.forward([[-2, -1, 0]])
    print(softmax.output)
    softmax.forward([[0.5, 1, 1.5]])
    print(softmax.output)

    X, y = spiral_data(samples = 100, classes = 3)
    dense1 = layer_dense(2, 3)
    activation1 = ReLU()
    dense2 = layer_dense(3, 3)
    activation2 = Activation_Softmax()
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output[:5])