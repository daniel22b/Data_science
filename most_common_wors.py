import sys, re
from collections import Counter
from typing import Type

# try:
#     num_words = int(sys.argv[1])
# except:
#     print("sposob uzycia: most_common_words.py liczba_slow")
#     sys.exit(1)

# counter = Counter(word.lower()
#                   for line in sys.stdin
#                   for word in line.strip().split() if word)

# for word, count in counter.most_common(num_words):
#     sys.stdout.write(str(count))
#     sys.stdout.write("\t")
#     sys.stdout.write(word)
#     sys.stdout.write("\n")

import re

# 1. Otwieranie pliku do odczytu (tryb 'r')
with open('dane.txt', 'r') as file_for_reading:
    data = file_for_reading.read()

# 2. Otwieranie pliku do zapisu (tryb 'w') — nadpisuje zawartość pliku
# with open('dane.txt', 'w') as file_for_writing:
#     file_for_writing.write("Nowa zawartość pliku\n")

# 3. Otwieranie pliku do dopisania (tryb 'a')
with open('dane.txt', 'a') as file_for_appending:
    file_for_appending.write("Dopisano kolejną linię\n")

# 4. Przetwarzanie danych z pliku
# with open('dane.txt', 'r') as f:
#     # Na przykład zliczanie linii zaczynających się od "#"
#     starts_with_hash = 0
#     for line in f:
#         if re.match(r"^#", line):
#             starts_with_hash += 1

# print(f"Liczba linii zaczynających się od '#': {starts_with_hash}")

def get_domain(email_adreess : str) -> str:
    return email_adreess.lower().split("@")[-1] if "@" in email_adreess else None

with open("dane.txt","r") as f:
    domain_counts = Counter(get_domain(word)
                            for line in f 
                            for word in line.split()
                            if "@" in word)

# Wyświetl wyniki
for domain, count in domain_counts.items():
    if domain:  # Ignoruj puste domeny
        print(f"{domain}: {count}")