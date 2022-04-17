import numpy as np
import nnfs

def ReLU_com_if(inputs):
    output = []

    for i in inputs:
        if i > 0:
            output.append(i)
        else:
            output.append(0)

    return output

def ReLU_com_max(inputs):
    output = []

    for i in inputs:
        output.append(max(0, i))
    
    return output

def ReLU_com_max_total(inputs):
    output = np.maximum(0, inputs)
    return output

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = ReLU_com_max(inputs)
print(output)
output = ReLU_com_if(inputs)
print(output)
output = ReLU_com_max_total(inputs)
print(output)