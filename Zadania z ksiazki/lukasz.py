# import statistics
# import math
# import random
# import numpy as np
# import matplotlib.pyplot as plt

# # Dane
# liczba_osob = range(1, 11)
# zlotych_w_portfelu = [x**2 for x in liczba_osob]

# # Obliczanie średniej i odchylenia standardowego
# srednia_portfela = statistics.mean(zlotych_w_portfelu)
# n = len(zlotych_w_portfelu)  # liczba elementów
# sigma = math.sqrt(sum([(xi - srednia_portfela) ** 2 for xi in zlotych_w_portfelu]) / n)

# # Tworzenie 35 próbek i obliczanie ich średnich
# lista_probek = [statistics.mean(random.choices(zlotych_w_portfelu, k=6)) for _ in range(35)]

# # Tworzenie wykresu
# fig, ax1 = plt.subplots()

# # Wykres słupkowy dla oryginalnych danych
# ax1.bar([x - 0.5 for x in liczba_osob], zlotych_w_portfelu, color='red', alpha=0.3, label='Złote w portfelu')
# ax1.set_xlabel("Liczba osób")
# ax1.set_ylabel("Złote w portfelu", color='red')
# ax1.tick_params(axis='y', labelcolor='red')

# # Tworzenie drugiej osi Y dla histogramu
# ax2 = ax1.twinx()

# # Histogram dla średnich z próbek
# ax2.hist(lista_probek, bins=10, alpha=0.7, color='blue', edgecolor='black', label='Średnie z próbek')
# ax2.set_ylabel("Częstotliwość", color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')

# # Dodanie legendy
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# # Wyświetlanie wykresu
# plt.show()

import random
import statistics
import matplotlib.pyplot as plt
import numpy as np

# Dane: liczba osób i ich portfele
liczba_osob = range(1, 11)
zlotych_w_portfelu = [x**2 for x in liczba_osob]  # np. portfel 1 osoby zawiera 1 zł, 2 osoba 4 zł itd.

# Tworzymy 35 próbek, każda po 6 elementów
lista_probek = [statistics.mean(random.choices(zlotych_w_portfelu, k=6)) for _ in range(35)]

# Rysowanie histogramu dla średnich z próbek
plt.hist(lista_probek, bins=10, alpha=0.7, color='blue', edgecolor='black', label='Średnie z próbek')

# Dodanie rozkładu normalnego na podstawie Centralnego Twierdzenia Granicznego
mu = statistics.mean(zlotych_w_portfelu)  # średnia dla całej populacji
sigma = statistics.stdev(zlotych_w_portfelu)  # odchylenie standardowe
n = 6  # wielkość próbki

# Wartości na osi X
xs = np.linspace(min(lista_probek), max(lista_probek), 100)

# Obliczanie teoretycznego rozkładu normalnego na podstawie CTG
ys = (1 / (sigma * np.sqrt(2 * np.pi / n))) * np.exp(-0.5 * ((xs - mu) ** 2 / (sigma ** 2 / n)))

# Dodanie wykresu teoretycznego rozkładu normalnego
plt.plot(xs, ys, label='Teoretyczny rozkład normalny', color='red')

# Tytuł i etykiety
plt.title('Rozkład średnich z próbek a rozkład normalny (CTG)')
plt.xlabel('Średnia z próbki')
plt.ylabel('Częstotliwość')
plt.legend()

# Wyświetlenie wykresu
plt.show()
