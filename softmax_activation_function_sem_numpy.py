import numpy as np
from nnfs.datasets import spiral_data
import nnfs

layer_outputs = [4.8, 1.21, 2.385]
E =  2.71828182846

exp_values = []

for output in layer_outputs:
    exp_values.append(E ** output)

print(f"exponentiated values: {exp_values}")

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(f"Normalized exponentiated values: {norm_values}")
print(f"Sum of normalized values: {sum(norm_values)}")