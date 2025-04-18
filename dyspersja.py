import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower

# Parametry testu
effect_size = 0.5  # rozmiar efektu (np. różnica w średnich)
alpha = 0.05  # poziom istotności
n = 50  # wielkość próby

# Tworzymy obiekt PowerAnalysis
analysis = TTestIndPower()

# Obliczamy moc testu
power = analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha)
print("Moc testu:", power)
