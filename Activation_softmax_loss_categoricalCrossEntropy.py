from cross_entropy_class import loss_categoricalCrossEntropy
import numpy as np

class Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        soma = np.sum(inputs, axis = 1, keepdims = True)
        probabilities_distribution = exp_inputs / soma
        self.output = probabilities_distribution

class Loss:
    def calculate(self, output, y_true):
        sample_losses = self.forward(output, y_pred)
        data_loss = np.mean(sample_losses)
        return data_loss

class Categorical_cross_entropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        samples = len(y_pred)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        
        negative_likelihoods = -np.log(correct_confidences)
        return(negative_likelihoods)

class Activation_Softmax_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs/samples