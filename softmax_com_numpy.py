import numpy as np

def metodo1_normalizacao(layer_outputs):
    exp_values = np.exp(layer_outputs)
    norm_values = exp_values / np.sum(exp_values)
    return norm_values

def metodo2_normalizacao(layer_outputs):
    exp_values = np.exp(layer_outputs)
    probabilities = exp_values / np.sum(exp_values, axis = 0, keepdims = True)
    return probabilities

def metodo3_normalizacao():
    layer_outputs = np.array([[4.8, 1.21, 2.385],
                              [8.9, -1.81, 0.2],
                              [1.41, 1.051, 0.026]])

    print(f"Sum without axis =  np.sum(layer_outputs)")
    print(f"This will be identical to the above since default is None: {np.sum(layer_outputs, axis = None)})")
    print(f"Axis equals to 0 means columns: {np.sum(layer_outputs, axis = 0)}")
    print(f"Axis equals to 1 means rows: {np.sum(layer_outputs, axis = 1)}")

if __name__ == '__main__':
    layer_outputs = [4.8, 1.21, 2.385]
    norm_values = metodo1_normalizacao(layer_outputs)
    probabilities = metodo2_normalizacao(layer_outputs)
    print(norm_values, probabilities)
    print(np.sum(norm_values), np.sum(probabilities))
    metodo3_normalizacao()