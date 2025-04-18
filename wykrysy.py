from matplotlib import pyplot as plt
import numpy as np
from collections import Counter


# years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
# gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# plt.plot(years, gdp, color = 'red', marker = 'o', linestyle = 'solid')

# plt.title("Normalny PKB")

# plt.ylabel("Mld dol.")
# plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# movies = ["Anndisa", "ghjdsvahj jj", "wjikqheik", "iiiuuuhg","ejkjjj uuu ii"]
# num_oscars = [3, 3, 7, 5, 2]

# plt.bar(range(len(movies)), num_oscars)
# plt.ylabel("liczba nagrod")
# plt.title("filmy")
# plt.xticks(range(len(movies)), movies)

# plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  
# grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]

# histogram = Counter(min(grade // 10 *10, 90)for grade in grades)

# plt.bar([x +5 for x in histogram.keys()],
#         histogram.values(),
#         10, edgecolor = 'black', linewidth = 0.3)
# plt.axis([-5,105,0,5])
# plt.xticks([10 * i for i in range(11)])
# plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# # Listy z tytułami filmów i liczbą nagród dla dwóch lat
# movies = ["Film A", "Film B", "Film C", "Film D"]
# oscars_2020 = [3, 5, 2, 4]   # liczba nagród w 2020 roku
# oscars_2021 = [4, 6, 3, 5]   # liczba nagród w 2021 roku

# # Pozycje na osi X (jeden indeks dla każdego filmu)
# x = np.arange(len(movies))

# # Szerokość słupka
# width = 0.35

# # Wykres dla pierwszej serii (2020)
# plt.bar(x - width/2, oscars_2020, width, label='2020')

# # Wykres dla drugiej serii (2021)
# plt.bar(x + width/2, oscars_2021, width, label='2021')

# # Etykiety i tytuły
# plt.xlabel("Filmy")
# plt.ylabel("Liczba nagród")
# plt.title("Liczba nagród dla filmów w latach 2020 i 2021")
# plt.xticks(x, movies)  # Ustawienie nazw filmów jako etykiet osi X
# plt.legend()

# plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# variance = [1,2,4,8,16,32,64,128,256]
# # bias_squarded = list(reversed(variance))
# bias_squarded = [256,128,64,32,16,8,4,2,1]

# total_error = [x + y for x,y in zip(variance, bias_squarded)]
# xs = [i for i,_ in enumerate(variance)]

# plt.plot(xs, variance, 'g-', label='variance')
# plt.plot(xs, bias_squarded, 'r-.', label='bias^2')
# plt.plot(xs, total_error, 'b:', label='total error')

# plt.legend(loc = 0)
# plt.xlabel("Stopien skomplikowania modelu")
# plt.title("Kompromies pomiedzy progowa i zlozonoscia modelu")
# # plt.axis([-5,10,-50,300])
# plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# friends = [70,65,72,63,71,64,60,64,67]
# minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
# labels = ['a','b','c','d','e','f','g','h','j']

# plt.scatter(friends, minutes,color = 'red')

# for label, friend_count, minute_count in zip(labels, friends, minutes):
#         plt.annotate(label,
#         xy = (friend_count, minute_count),
#         xytext = (5, -5),
#         textcoords = 'offset points')

# plt.title("Czas spedzony na stronie a liczba znajomych")
# plt.xlabel("liczba znajomych")
# plt.ylabel("Dzienny czas spedzony na stronie (minuty)")
# plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# test_1_grades = [99, 90, 85, 97, 80]
# test_2_grades = [100, 85, 60, 90, 70]

# plt.scatter(test_1_grades, test_2_grades)
# plt.title("Abc")
# plt.xlabel(test_1_grades)
# plt.ylabel(test_2_grades)
# plt.axis([60,110,40,110])
# plt.xticks(np.arange(60,111, 5))
# plt.xticks(test_1_grades, test_1_grades, color ='red')
# # plt.xticks(np.arange(60,111, 5))
# plt.show()



# import statistics
# import math
# import random
# import numpy as np
# liczba_osob = range(1,11)
# zlotych_w_portfelu = [x**2 for x in liczba_osob]

# srednia_portfela = statistics.mean(zlotych_w_portfelu)
# n = len(zlotych_w_portfelu)

# sigma = math.sqrt(sum([(xi - srednia_portfela) **2 for xi in     zlotych_w_portfelu])/n) 


# lista_probek = [statistics.mean(random.choices(zlotych_w_portfelu, k=6)) for _ in range(35)]

# plt.hist(lista_probek, bins=10,alpha =0.7,color='blue')


# plt.bar([x -0.5 for x in liczba_osob], zlotych_w_portfelu,
#         color='red',
#         alpha = 0.3)

# plt.xlabel("liczba osob")
# plt.ylabel("zlotych w portfelu")
# plt.xticks(liczba_osob)

# plt.show()

from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float,float]:
    mi = p * n
    sigma = math.sqrt(p * (1-p)* n)
    return mi, sigma

# from scratch.probability import normal_cdf

normal_propabilty_below = normal_cdf

def normal_propability_above(lo: float,
                             mi: float = 0,
                             sigma: float = 1) -> float:
    return 1 - normal_cdf(lo, mi, sigma)

def normal_propability_between(lo: float,
                                hi: float,
                                mi: float = 0,
                                sigma: float = 1) -> float:
    return normal_cdf(hi, mi, sigma) - normal_cdf(lo, mi, sigma)

def normal_propability_outside(lo: float,
                               hi: float,
                               mi: float = 0,
                               sigma: float = 1) -> float:
    return 1 - normal_propability_between(lo, hi, mi, sigma)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# from scratch.probability import inverse_normal_cdf

# def normal_upper_bound(propability: float,
#                        mi: float = 0,
#                        sigma: float = 1)-> float:
#     return inverse_normal_cdf(propability, mi, sigma)

# def normal_lower_bound(propability: float,
#                        mi: float = 0,
#                        sigma: float = 1)-> float:
#     return inverse_normal_cdf(1 - propability, mi, sigma)

# def normal_two_sided_bounds(propability: float,
#                        mi: float = 0,
#                        sigma: float = 1)-> Tuple[float,float]:
#     tail_propability = (1- propability) /2

#     uppder_bound = normal_lower_bound(tail_propability, mi, sigma)

#     lower_bound = normal_upper_bound(tail_propability, mi, sigma)

#     return lower_bound, uppder_bound

# mi_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

# lower_bound, upper_bound = normal_two_sided_bounds(0.95, mi_0, sigma_0)

# lo, hi = normal_two_sided_bounds(0.95, mi_0, sigma_0)

# mi_1,sigma_1 = normal_approximation_to_binomial(1000, 0.5)

# type_2_propabilty  = normal_propability_between(lo,hi,mi_1,sigma_1)
# power = 1 - type_2_propabilty

# hi = normal_upper_bound(0.95,mi_0, sigma_0)

# type_2_propabilty = normal_propabilty_below(hi,mi_1, sigma_1)
# power = 1 - type_2_propabilty


    