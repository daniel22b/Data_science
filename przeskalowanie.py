from typing import Tuple, List
from scratch.linear_algebra import vector_mean, Vector
from scratch.statistics import standard_deviation
import numpy as np

def scale(data: List[Vector])-> Tuple[Vector, Vector]:
    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [ standard_deviation([vector[i] for vector in data])
              for i in range(dim)]
    
    return means, stdevs
vectors = [[-3,-1,1],[-1,0,1],[1,1,1]]

means, stdevs = scale(vectors)

def rescale(data: List[Vector]) -> List[Vector]:

    dim = len(data[0])
    means , stdevs = scale(data)
    rescaled = [v[:] for v in data]

    for v in rescaled:
        for i in range(dim):
            if stdevs[i] >0:
                v[i] = (v[i]- means[i])/ stdevs[i]
    
    return rescaled

print(rescale(vectors))

#LEPSZY SPOSOB
import numpy as np

def scale(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.mean(data, axis=0)  
    stdevs = np.std(data, axis=0)  
    return means, stdevs

def rescale(data: np.ndarray) -> np.ndarray:
    means, stdevs = scale(data)
    stdevs = np.where(stdevs == 0, 1, stdevs)
    return (data - means) / stdevs


vectors = np.array([[-3, -1, 1], [-1, 2, 1], [1, 1, 3]])

means, stdevs = scale(vectors)
print()
print("Means:", means)
print("Standard Deviations:", stdevs)

rescaled = rescale(vectors)
print("Rescaled Vectors:\n", rescaled)

mean = np.mean(rescaled, axis=0)
st = np.std(rescaled, axis=0)
print(np.isclose(mean,0), st)