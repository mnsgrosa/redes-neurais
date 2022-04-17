import numpy as np
class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
       return np.dot(inputs, self.weights)


inputs = [0.5, 2, 4, 3]

l = layer_dense(4, 3)
print(l.forward(inputs))
print(np.random.randn(2, 5))
print(np.zeros((4, 4)))