from scratch.statistics import de_mean
from scratch.linear_algebra import Vector
from typing import Tuple
from scratch.statistics import correlation, standard_deviation, mean, num_friends_good,daily_minutes_good
import matplotlib.pyplot as plt
from scratch.statistics import de_mean
import numpy as np

#wZOR NA REGRESJE LINIOWA
def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta*x_i + alpha

def predict2(alpha:float, beta:float, x:float) -> float:
    return np.dot([1,x],[alpha,beta])

#BLAD PREDYKCJI 
def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    return predict(alpha, beta, x_i) - y_i

#SUMA BLEDOW RESZTOSWYCH (SSE)
def sum_of_sqerrors(alpha: float, beta: float, x:Vector, y:Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x,y))

# OBLICZENEI NACHYLENIA I WYRAZU WOLNEGO 
def least_squares_fit(x: Vector, y:Vector) -> Tuple[float,float]:
    beta = correlation(x,y) * standard_deviation(y) /standard_deviation(x)
    alpha = mean(y) - beta *mean(x)
    return alpha, beta

alpha, beta = least_squares_fit(num_friends_good,daily_minutes_good)

# n = len(num_friends_good)
# print(alpha, beta)
# print(sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)/n)

# WYKRES REGRESJI 
ys = [predict2(alpha, beta, x_i) for x_i in num_friends_good]
plt.scatter(num_friends_good, daily_minutes_good)
plt.plot(num_friends_good, ys, color = 'red')
plt.show()

#CALKOWITA SUMA KWADRATOW (TSS)
def total_sum_of_squares(y:Vector) -> float:
    return sum(v **2 for v in de_mean(y))


#WSPOLCZYNNIK DETERMINACJI R2
def r_squared(alpha: float, beta:float, x:Vector, y:Vector)-> float:
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y)/total_sum_of_squares(y))


rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
print(f"rsq: {rsq}")
x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x ]

assert least_squares_fit(x,y) == (-5, 3)

print(least_squares_fit(num_friends_good, daily_minutes_good))

print(f"Pedykcja 1: {predict(alpha, beta, 3)}")
print(f"Pedykcja 2: {predict2(alpha, beta, 3)}")


#OBLICZANIE ALPHA I BETA ZA POMOCA GRADIENTU 
import random 

from scratch.gradient_descent import gradient_step

num_epochs = 10000
random.seed(0)

guess = [ random.random(), random.random()]

learning_rate = 0.00001

for _ in range(num_epochs):
    alpha, beta = guess


    grad_a = sum(2 * error(alpha, beta,x_i, y_i) for x_i, y_i in zip(num_friends_good, daily_minutes_good))

    grad_b = sum(2 * error(alpha, beta,x_i, y_i) * x_i for x_i, y_i in zip(num_friends_good, daily_minutes_good))

    loss = sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)

            

    guess = gradient_step(guess, [grad_a, grad_b], - learning_rate)
print(f"guess {guess}")



#Regresja wieloraka
from scratch.linear_algebra import dot, Vector
from typing import List

def predict(x: Vector, beta: Vector)-> float:
    return dot(x, beta)

def error(x:Vector, y:float, beta:Vector)->float:
    return predict(x, beta) - y

def squared_error(x: Vector, y:float, beta:Vector)->float:
    return error(x,y,beta) **2

x = [1,2,3]
y = 30

beta = [4,4,4]

assert squared_error(x,y,beta) == 36

def sqerror_gradient(x:Vector, y:float, beta:Vector)-> Vector:
    err = error(x,y,beta)
    return [2 * err * x_i for x_i in x]

print(sqerror_gradient(x,y,beta))

import random
import tqdm
from scratch.linear_algebra import vector_mean
from scratch.gradient_descent import gradient_step

def least_squares_fit(xs: List[Vector], 
                      ys: List[float],
                      learning_rate: float = 0.0001,
                      num_steps:int = 1000,
                      batch_size: int = 1) ->Vector:
    
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_gradient(x,y, guess)
                                    for x, y in zip(batch_xs, batch_ys)])
            
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess
    

from scratch.statistics import daily_minutes_good
from scratch.gradient_descent import gradient_step

random.seed(0)

learning_rate = 0.001
inputs =[
    [2, 30, 7],   
    [5, 22, 6],
    [1, 40, 8],
    [4, 35, 5],
    [3, 28, 7]
]

y = [100, 80, 130, 90, 95]
beta = least_squares_fit(inputs, y, learning_rate,5000, 25 )

# print(beta)


from typing import TypeVar, Callable

X = TypeVar('X')
Stat = TypeVar('Start')

def bootstrap_sample(data: List[X]) -> List[X]:
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data: List[X],
                        stats_fn: Callable[[List[X]], Stat],
                        num_samples: int) -> List[Stat]:
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]