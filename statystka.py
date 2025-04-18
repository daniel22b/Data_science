from collections import Counter
import matplotlib.pyplot as plt
from typing import List

num_firends = [2,1,2,3,3,3]
friend_count = Counter(num_firends)

xs = range(20)
ys = [friend_count[x] for x in xs]
# plt.bar(xs,ys)
# plt.axis([0,20,0,25])
# plt.show()

def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs)

# print(mean(num_firends))

def _median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs)//2]

def _median_even(xs: List[float]) -> float:
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1]+ sorted_xs[hi_midpoint]) /2

def median(v: List[float]) -> float:
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

def quantile(x: List[float], p: float) -> float:
    p_index = int(p * len(x))
    return sorted(x)[p_index]

def mode(x: List[float]) -> List[float]:
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]

print(quantile(num_firends, 0.20))

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

mi = 4
sigma = 3

x = np.linspace(mi - 4*sigma, mi + 4*sigma, 1000)
pdf = stats.norm.pdf(x, mi, sigma)
cdf = stats.norm.cdf(x, mi, sigma)

plt.plot(x,pdf,label ="PDF", color = "blue")
plt.plot(x,cdf,label ="CDF", color = "red")

plt.axvline(mi, color='red', linestyle='--', label='Średnia (mu)')
plt.show()