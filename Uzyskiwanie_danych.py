import sys, re
from collections import Counter

try:
    num_words = int(sys.argv[1])
except:
    print("sposob uzycia: most_common_words.py liczba_slow")
    sys.exit(1)

counter = Counter(word.lower()
                  for line in sys.stdin
                  for word in line.strip().split() if word)

for word, count in counter.most_common(num_words):
    sys.stdout(str(count))
    sys.stdout("\t")
    sys.stdout(word)
    sys.stdout("\n")



regex = sys.argv[1]
count = 0

for line in sys.stdin:
    if re.search(regex, line):
        sys.stdout.write(line)

for x in sys.stdin:
    count += 1
print(count)

