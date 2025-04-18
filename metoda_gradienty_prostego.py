# from scratch.linear_algebra import Vector, dot
from typing import Callable
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

def sum_of_squers(v: Vector) -> float:
    return dot(v,v)

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:

    return(f(x+h)- f(x)) /h

def square(x :float) -> float:
    return x *x

def derivative(x:float)-> float:
    return 2*x

xs = range(-10,11)
actuals = [derivative(x) for x in xs]
estimats = [difference_quotient(square, x, h = 0.001) for x in xs]

plt.title("Rzeczywiste pochodne i ich szacunkowe wartosci")
plt.plot(xs, actuals, 'rx', label= "warosc rzeczywista")
plt.plot(xs, estimats, 'b+', label='warosc oszacowana')
plt.legend(loc=9)
plt.grid()
# plt.show()

# def f(x:float) -> float:
#     return ((3*x)**2) +(3*x) + 5

# def pochodna(f, x,h=000.1):
#     return (f(x + h) - f(x)) / h

# xss = np.linspace(-10, 10, 100)

# f_values = f(xss)
# f_prime_values = [pochodna(f, x)for x in xss]

# plt.plot(xss, f_values, label='f(x) = (3x)^2 + 3x + 5', color='blue')
# plt.plot(xss, f_prime_values, label="Pochodna f'(x)", linestyle='dashed', color='red')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Wykres funkcji i jej pochodnej')
# plt.grid(True)
# plt.show()


def partial_difference_quotient(f:Callable[[Vector], float],
                                v:Vector,
                                i: int,
                                h:float)->float:
    
    w = [v_j +(h if j == i else 0)
        for j, v_j in enumerate(v)]
    
    return (f(w) - f(v)) / h

def estimate_difference_quotient(f:Callable[[Vector], float],
                                v:Vector,
                                i: int,
                                h:float)->float:
    
    return [partial_difference_quotient(f,v,i,h) for i in range(len(v))]
import random
# from scratch.linear_algebra import distance, add, scalar_multiply, vector_mean

def gradient_step(v:Vector, gradient: Vector, step_size: float)-> Vector:
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v:Vector) -> Vector:
    return [2 * v_i for v_i in v]

v = [random.uniform(-10,10)for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v)
    v = gradient_step(v,grad, -0.01)
    print(epoch, v)

assert distance(v, [0,0,0]) < 0.001

inputs = [(x,20*x+5) for x in range(-50,50)]

def linear_gradient(x:float, y:float, theta: Vector) -> Vector:
    m, b = theta
    predicted = m * x + b
    error = (predicted - y)
    squared_error = error **2
    grad = [2*error*x, 2*error]

    return grad

theta = [random.uniform(-1,1), random.uniform(-1,1)]
learning_rate = 0.001
for epoch in range(5000):
    grad = vector_mean([linear_gradient(x,y,theta) for x,y in inputs])
    theta = gradient_step(theta,grad, -learning_rate)
    print(epoch, theta)

m, b = theta
assert 19.9 <m<20.1 ,"wartosc m powinna wynosic okolo 20"
assert 4.9 <b<5.1 ," wartosc b powinna wynosic okolo 5 "
#"_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________"

import numpy as np
import matplotlib.pyplot as plt

# Dane
x = np.array(range(-50, 50))
y = 20 * x + 5  # Równanie rzeczywistej linii z szumem

# Obliczanie współczynników regresji liniowej
A = np.vstack([x, np.ones(len(x))]).T  # Tworzymy macierz [x | 1]
slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]  # Rozwiązujemy równanie Ax = b

print(f"Slope: {slope}, Intercept: {intercept}")

# Rysowanie wyników
plt.scatter(x, y, label="Dane")
plt.plot(x, slope * x + intercept, color="red", label="Linia regresji")
plt.legend()
plt.show()
#_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Dane
x = np.array(range(-50, 50)).reshape(-1, 1)  # x jako wektor kolumnowy
y = 20 * x + 5  # Równanie rzeczywistej linii

# Model regresji liniowej
model = LinearRegression()
model.fit(x, y)  # Dopasowanie modelu

# Współczynniki regresji
slope = model.coef_[0][0]  # Nachylenie
intercept = model.intercept_[0]  # Punkt przecięcia
print(f"Slope: {slope}, Intercept: {intercept}")

# Rysowanie wyników
plt.scatter(x, y, label="Dane")
plt.plot(x, model.predict(x), color="red", label="Linia regresji")
plt.legend()
plt.show()

from typing import TypeVar, List, Iterator

T = TypeVar('T')

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle:random.shuffle(batch_starts)

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]
dataset = [random.randint(10) for _ in range(100)]
print(dataset)
batches = minibatches()



