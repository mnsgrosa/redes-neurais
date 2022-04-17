import numpy as np

a = [1, 2, 3]
print(np.array(a))
print(np.expand_dims(np.array(a), axis = 0))
#uma maneira de transpor a matriz
print(np.expand_dims(np.array(a), axis = 1))
#outra maneira de transpor a matriz
print(np.array([a]).T)

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
b = np.array([b]).T

print(np.dot(a, b))