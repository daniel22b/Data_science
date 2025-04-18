from typing import TypeVar, List, Iterator, Dict
import random
from collections import Counter
# from scratch.linear_algebra import Vector
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sympy import symbols, diff

# T = TypeVar('T')

# def minibatches(dataset: List[T],
#                 batch_size: int,
#                 shuffle: bool = True) -> Iterator[List[T]]:
    
#     batch_starts = [start for start in range(0, len(dataset), batch_size)]

#     if shuffle:random.shuffle(batch_starts)

#     for start in batch_starts:
#         end = start + batch_size
#         yield dataset[start:end]

# random.seed(0)
# dataset =[random.randint(0,10) for _ in range(100)]

# batches = minibatches(dataset, 10)
# for batch in batches:
#     print(batch)

T = TypeVar("T")

transactions = [
    {"product_id": 1, "quantity": 10, "price": 20.5, "date": "2023-01-15"},
    {"product_id": 2, "quantity": 5, "price": 15.0, "date": "2023-01-16"},
    {"product_id": 8, "quantity": 8, "price": 20.5, "date": "2023-02-10"},
    {"product_id": 9, "quantity": 12, "price": 7.25, "date": "2023-02-12"},
    {"product_id": 2, "quantity": 10, "price": 15.0, "date": "2023-03-03"},
    {"product_id": 1, "quantity": 5, "price": 20.5, "date": "2023-03-05"},{"product_id": 4, "quantity": 10, "price": 20.5, "date": "2023-01-15"},
    {"product_id": 5, "quantity": 5, "price": 15.0, "date": "2023-01-16"},
    {"product_id": 9, "quantity": 8, "price": 20.5, "date": "2023-02-10"},
    {"product_id": 3, "quantity": 12, "price": 7.25, "date": "2023-02-12"},
    {"product_id": 8, "quantity": 10, "price": 15.0, "date": "2023-03-03"},
    {"product_id": 1, "quantity": 5, "price": 20.5, "date": "2023-03-05"},{"product_id": 7, "quantity": 10, "price": 20.5, "date": "2023-01-15"},
    {"product_id": 6, "quantity": 5, "price": 15.0, "date": "2023-01-16"},
    {"product_id": 8, "quantity": 8, "price": 20.5, "date": "2023-02-10"},
    {"product_id": 7, "quantity": 12, "price": 7.25, "date": "2023-02-12"},
    {"product_id":6, "quantity": 10, "price": 15.0, "date": "2023-03-03"},
    {"product_id": 9, "quantity": 5, "price": 20.5, "date": "2023-03-05"},{"product_id": 4, "quantity": 10, "price": 20.5, "date": "2023-01-15"},
    {"product_id": 5, "quantity": 5, "price": 15.0, "date": "2023-01-16"},
    {"product_id": 6, "quantity": 8, "price": 20.5, "date": "2023-02-10"},
    {"product_id": 3, "quantity": 12, "price": 7.25, "date": "2023-02-12"},
    {"product_id": 8, "quantity": 10, "price": 15.0, "date": "2023-03-03"},
    {"product_id": 7, "quantity": 5, "price": 20.5, "date": "2023-03-05"},
]

            



total_quantities ={}
for transatcion in transactions:
    product_id = transatcion["product_id"]
    quantity = transatcion["quantity"]

    if product_id in total_quantities:
        total_quantities[product_id] += quantity
    else: 
        total_quantities[product_id] = quantity


sorted_dict = dict(sorted(total_quantities.items(), key=lambda x:x[0]))

data_x = []
data_y = []

for key, value in sorted_dict.items():
    if key not in data_x:
        data_x.append(key)
        data_y.append(value)

xs = np.array(data_x).reshape(-1,1)
ys = np.array(data_y)

model = LinearRegression()
model.fit(xs, ys)
priction = model.predict(xs)

# new_data = np.array(data_x)
# A = np.vstack((new_data, np.ones(len(data_x)))).T
# slope ,intercep = np.linalg.lstsq(A, data_y, rcond=None)[0]


plt.scatter(xs, ys, color="red", label="Dane")
plt.plot(xs, priction, color="green", label=" Linia regresji")
plt.xlabel("Produkty ID")
plt.ylabel("Calkowita ilosc")
plt.legend()
plt.show()
m = model.coef_[0]
b = model.intercept_

def f(x):
    return m*x + b


print("Współczynnik kierunkowy (m):", model.coef_[0])
print("Wyraz wolny (b):", model.intercept_)
print()
# print(slope)
# print(intercep)

# random.seed(4)
# theta = [random.uniform(-1,1), random.uniform(-1,1)]
# def gradient1(x:int,y:int, theta: Vector)->Vector:
#     m, b = theta
#     predict = m*x + b
#     error = predict - y
#     sqared_error = error**2
#     grad = [2 *error * x, 2*error]

#     return grad

# def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
#     step = [step_size * g for g in gradient]
#     return [v_i - s for v_i, s in zip(v, step)]

# learning_rate = 0.0001   
# iterations = 5000

# for epoch in range(iterations):
#     grad = [gradient1(x, y, theta) for x, y in zip(xs, data_y)]
#     mean_grad = [sum(g) / len(g) for g in zip(*grad)]  
#     theta = gradient_step(theta, mean_grad, learning_rate)

#     if epoch % 500 == 0:  
#         print(f"Epoch {epoch}, theta: {theta}")


# # Rysowanie wykresu
# plt.figure(figsize=(10, 6))

# # Rysowanie punktów danych
# plt.scatter(xs, data_y, color="red", label="Dane (x, y)")

# # Generowanie punktów do linii regresji
# y_pred = [theta[0] * x + theta[1] for x in xs]

# # Rysowanie linii regresji
# plt.plot(xs, y_pred, color="blue", label="Linia regresji")

# # Ustawienia wykresu
# plt.title("Wizualizacja regresji liniowej i punktów danych")
# plt.xlabel("Produkt ID")
# plt.ylabel("Całkowita ilość")
# plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
# plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
# plt.legend()
# plt.grid()
# plt.show()
