import matplotlib.pyplot as plt
import numpy as np
import random

objects = 10
salary = [np.random.randint(10000,20000) for _ in range(objects)]

# plt.figure()

# plt.hist(salary, bins=20, color='red', edgecolor='black', alpha=0.7)
xs = np.arange(-10,10,0.1)
ys = [(x**2) * np.sin(x) for x in xs]
ys2 = [(x**2) * np.cos(x) for x in xs]


plt.plot(xs, ys, color='red')
plt.plot(xs,ys2, color= 'blue')
plt.show()
