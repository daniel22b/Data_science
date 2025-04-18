from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
import re
import math
from collections import defaultdict
from io import BytesIO
import requests
import tarfile
import glob, re
from scratch.machine_learning import split_data
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def tokenize(txt:str) -> Set[str]:
    text = txt.lower()
    all_words = re.findall("[a-z0-9]+", text)
    return set(all_words)

assert tokenize("Dzisiaj jest 08.04.2025") == {"dzisiaj", "jest", "08", "04", "2025"}

class Message(NamedTuple):
    text:str
    is_spam: bool

class NaiveBayesClassifier:
    def __init__(self, k:float = 0.5) -> None:
        self.k = k

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0 

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages +=1

            else:
                self.ham_messages +=1
            
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probablities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text:str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probablities(token)

            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam) 
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam) 
        prob_if_ham = math.exp(log_prob_if_ham)

        return prob_if_spam/(prob_if_spam + prob_if_ham) 
    

messages = [Message("spAm rules", is_spam=True),
            Message("ham RUles", is_spam=False),
            Message("hELlo ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam" : 1 , "rules" : 1}
assert model.token_ham_counts == {"ham" : 2 , "rules" : 1, "hello":1}


text = "hello spam"

probs_if_spam = [
    (1 + 0.5) / (1 + 2*0.5),
    1 - (0 + 0.5) / (1 + 2*0.5),
    1 - (1 + 0.5) / (1 + 2*0.5),
    (0 + 0.5)/ (1 + 2*0.5)
]
probs_if_ham = [
    (0 + 0.5) / (2 + 2*0.5),
    1 - (2 + 0.5) / (2 + 2*0.5),
    1 - (1 + 0.5) / (2 + 2*0.5),
    (1 + 0.5)/ (2 + 2*0.5)
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

assert model.predict(text) == p_if_spam/(p_if_spam + p_if_ham)

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2",
         ]

OUTPUT_DIR = 'spam-data'

for filename in FILES:
    content = requests.get(f"{BASE_URL}/{filename}").content

    fin = BytesIO(content)

    with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR, filter="data")

path ='spam-data/*/*'
data: List[Message] = []

for filename in glob.glob(path):
    is_spam = "ham" not in filename

    with open(filename, errors='ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject: "):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject, is_spam))
                break
texts, labels = zip(*data)


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25,random_state=12 )

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print(classification_report(y_test,y_pred))

word_probs = model.feature_log_prob_[1]
words = vectorizer.get_feature_names_out()

sorted_words = sorted(zip(words,word_probs), key=lambda x: x[1], reverse=True )

print("Najbardziej charakterystyczne słowa dla spamu:")
for word, prob in sorted_words[:10]:
    print(f"{word}: {math.exp(prob)}")



# random.seed(0)
# train_messages, test_messages, = split_data(data, 0.75)

# model = NaiveBayesClassifier()
# model.train(train_messages)


# from collections import Counter

# predictions = [(message,model.predict(message.text))
#                 for message in test_messages]


# confusion_matrix = Counter((message.is_spam, spam_propability > 0.5)
#                            for message, spam_propability in predictions)

# # print(confusion_matrix)

# def p_spam_given_token(token: str, model: NaiveBayesClassifier) ->float:

#     prob_if_spam, prob_if_ham = model._propablities(token)

#     return prob_if_spam/ (prob_if_spam + prob_if_ham)

# words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

# # print("spammiest_words", words[-10:])
# # print("hammiest_words", words[:10])

