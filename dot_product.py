import numpy as np

def sem_numpy_com_loop(inputs, weights, bias):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]
    
    output += bias
    print(output)

def com_numpy(inputs, weights, bias):
    output = np.dot(weights, inputs) + bias
    print(output)
    return output

if __name__ == '__main__':
    inputs = [1, 2, 3, 2.5]
    weights = [0.8, 0.2, -0.26, 4]
    bias = 2

    output = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + (inputs[3] * weights[3]) + bias
             
    print(output)
    sem_numpy_com_loop(inputs, weights, bias)
    com_numpy(inputs, weights, bias)