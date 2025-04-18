import numpy as np

matrix = np.array(np.random.randint(1,10,(5,5)))
mean = np.mean(matrix)
std = np.std(matrix)

matrix_norm = (matrix - mean)/std
x = matrix_norm.std()
print(np.round(x))