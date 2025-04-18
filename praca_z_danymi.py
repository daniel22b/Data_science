from typing import List, Dict
from collections import Counter
import math
import matplotlib.pyplot as plt
from scipy.stats import norm  
import random
from scratch.linear_algebra import Matrix,Vector,make_matrix
from scratch.statistics import correlation
import numpy as np



# Funkcja dla określenia rozmiaru "buckets"
def bucketsize(point: float, bucket_size: float) -> float:
    return bucket_size * math.floor(point / bucket_size)

# Funkcja do generowania histogramu
def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    return Counter(bucketsize(point, bucket_size) for point in points)

# Funkcja do rysowania histogramu
def plot_histogram(points: List[float], bucket_size: float, title: str = ''):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    # plt.show()


random.seed(0)
uniform = [200 * random.random() - 100 for _ in range(10000)]


normal = [57 * norm.ppf(random.random()) for _ in range(10000)]
x = [1,2,3,4,5,6,7,8,9,4,6,2,3,4,5,2,7,5,4]
# print(make_histogram(x,3))
# plot_histogram(x, 3,"Moj histogram")

# plot_histogram(uniform, 10, "Histogram rozkładu jednostajnego")

# plot_histogram(normal, 10, "Histogram rozkładu normalnego")

def random_normal()->float:
    return norm.ppf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [x + random_normal()/2 for x in xs]
ys2 = [-x + random_normal()/2 for x in xs]

plt.scatter(xs,ys1, color = 'red')
plt.scatter(xs,ys2, color= 'blue')
plt.show()

print(correlation(xs, ys1))
print(correlation(xs, ys2))


# def correlation_matrix(data: List[Vector]) ->Matrix:
#     def correlation_ij(i:int, j:int)->float:

#         return correlation(data[i], data[j])

#     return make_matrix(len(data), len(data), correlation_ij)


# data = [[1,2,3], [2,4,6], [3,6,9]]
# corr_matrix = correlation_matrix(data)
# x = np.array(corr_matrix)
# print(x)

# # FUNKCJA ZAGNIEZDZONA PRZYKLAD
# def sum(list:List[int])->int:
#     def plus(a,b):
#         return a + b
    
#     resuly = 0
#     for num in list:
#         resuly = plus(resuly, num)

#     return resuly

# sumof = [2,3,5,3,2,1]
# print(sum(sumof))

import pandas as pd
# data = np.random.rand(5,3)
# df = pd.DataFrame(data, columns=['v1', 'v2', 'v3'])

# correlation_matrix_pandas = df.corr()
# print(correlation_matrix_pandas)
#---------------------------------------------------------------------------



# corr_datax = np.random.rand(3,3)

# df = pd.DataFrame(corr_datax)
# corr_data = df.corr()
# print(corr_data)
# num_vectors = len(corr_data)

# x_min = np.min(corr_data)
# x_max = np.max(corr_data)
# y_min = np.min(corr_data)
# y_max = np.max(corr_data)

# fig ,ax = plt.subplots(num_vectors, num_vectors)

# for i in range(num_vectors):
#     for j in range(num_vectors):
#         if i != j:
#             ax[i][j].scatter(corr_data[j], corr_data[i])
            
#         else:
#             ax[i][j].annotate("seria"+ str(i), (0.5,0.5),
#         xycoords ='axes fraction',
#         ha ="center", va="center")

#         if i <num_vectors -1:
#              ax[i][j].xaxis.set_visible(False)
#         if j >0:
#              ax[i][j].yaxis.set_visible(False)

# ax[i][j].set_xlim(x_min+0.2, x_max+0.2)
# ax[i][j].set_ylim(y_min+0.2, y_max+0.2)

# ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
# ax[0][0].set_ylim(ax[0][1].get_ylim())
# plt.show()

random.seed(1)
data = np.random.rand(3,3)
vector = len(data)
fig, ax = plt.subplots(vector, vector)


xmin = np.min(data) 
xmax = np.max(data)
ymin = np.min(data)
ymax = np.max(data)

for i in range(vector):
    for j in range(vector):
        if i !=j :
            ax[i][j].scatter(data[j], data[i])
        else:
            ax[i][j].annotate(f"xyz",(0.5,0.5), xycoords = 'axes fraction')

        if i < vector - 1:
            ax[i][j].xaxis.set_visible(False)
        if j >0:
            ax[i][j].yaxis.set_visible(False)

ax[i][j].set_xlim(xmin, xmax)
ax[i][j].set_xlim(ymin, ymax)
plt.show()
