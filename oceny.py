import enum
import random

class Kid(enum.Enum):
    BOY = 0
    GIRL = 1

    @staticmethod
    def random_kid() -> "Kid":
        return random.choice([Kid.BOY, Kid.GIRL])


# Statystyki
both_girls = 0
older_girl = 0
either_girl = 0

# Ustawienie ziarna dla powtarzalności
random.seed(0)

# Symulacja
for _ in range(10000):
    younger = Kid.random_kid()
    older = Kid.random_kid()
    
    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

# Wyniki
print("Prawdopodobieństwo obu dziewczynek:", both_girls / either_girl)
print("Prawdopodobieństwo starszej dziewczynki:", older_girl / 10000)
print("Prawdopodobieństwo przynajmniej jednej dziewczynki:", either_girl / 10000)





