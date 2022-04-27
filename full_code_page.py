import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class softmax:
    def forward(self, inputs):
        print(np.max(inputs, axis = 1, keepdims = True))
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities

class loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = np.mean(sample_loss)
        return data_loss

class categorical_cross_entropy(loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[samples, y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

        negative_log_likehoods = -np.log(correct_confidences)
        return negative_log_likehoods

if __name__ == "__main__":
    X, y = spiral_data(samples = 100, classes = 3)
    dense1 = layer_dense(2, 3)
    activation1 = ReLU()
    dense2 = layer_dense(3, 3)
    activation2 = softmax()
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    print(activation2.output[:5])
    loss1 = categorical_cross_entropy()
    loss1.calculate(activation2.output, y)
    print(loss1) 