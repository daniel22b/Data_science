import matplotlib.pyplot as plt
import numpy as np
import random

# emplloyment_time = [user for user in range(1,21)]
# salary = [random.randint(100,10000)for _ in range(20)]

# emplloyment_time_mean = np.mean(emplloyment_time)
# salary_mean = np.mean(salary)

# numerator = sum((x_i - emplloyment_time_mean) * (y_i - salary_mean) for x_i, y_i in zip(emplloyment_time, salary))
# denominator = sum((x_i - emplloyment_time_mean) ** 2 for x_i in emplloyment_time)

# m = numerator/denominator

# b = salary_mean - m * emplloyment_time_mean
# regresja = [m * x + b for x in emplloyment_time]
# plt.plot(emplloyment_time,salary)
# plt.plot(emplloyment_time, regresja,color ="red")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# # Dane
# x = np.array(range(-50, 50))
# y = 20 * x + 5  # Równanie rzeczywistej linii z szumem

# # Obliczanie współczynników regresji liniowej
# A = np.vstack([x, np.ones(len(x))]).T  # Tworzymy macierz [x | 1]

# slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]  # Rozwiązujemy równanie Ax = b


# print(f"Slope: {slope}, Intercept: {intercept}")

# # Rysowanie wyników
# plt.scatter(x, y, label="Dane")
# plt.plot(x, slope * x + intercept, color="red", label="Linia regresji")
# plt.legend()
# plt.show()
 
import numpy as np

# Dane: x - godziny pracy, y - wynagrodzenie
x = np.arange(1,21).tolist()
y = [random.randint(1,20) for _ in range(20)]

# Tworzymy macierz A, która zawiera x i kolumnę jedynek (dla wyrazu wolnego)
A = np.vstack([x, np.ones(len(x))]).T

# Obliczamy nachylenie (slope) i wyraz wolny (intercept)
slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
ys = [slope * i + intercept for i in x]

print(f"Nachylenie: {slope}")
print(f"Wyraz wolny: {intercept}")


x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = sum((x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(x, y))
denominator = sum((x_i - x_mean) ** 2 for x_i in x)

m = numerator/denominator

b = y_mean - m * x_mean
regresja = [m * x + b for x in x]

plt.plot(x,y)
plt.plot(x,ys, color = "red",ls='--')
plt.plot(x,regresja,color= "blue",alpha =0.4)

plt.show()

