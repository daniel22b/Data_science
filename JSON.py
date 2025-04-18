# import json
# from bs4 import BeautifulSoup
# import requests
# from collections.abc import Callable 

# url = ("https://webscraper.io/test-sites")

# html = requests.get(url).text
# soup = BeautifulSoup(html, 'html.parser')

# # print(soup.prettify())

# important_paragraphs = soup.find_all('p')
# span_texts = [p.text.strip() for p in important_paragraphs]
# for x in span_texts[:4]:
#     print(x)


# serialized = """{
#     "title": "Data Science Book",
#     "author": "Joel Grus",
#     "PublicationYear": 2019,
#     "topics": ["data", "science", "data science"]
# }"""


# deserialized = json.loads(serialized)
# x = deserialized["PublicationYear"]
# print(x)

# github_user = "joelgrus"
# endpoint = f"https://api.github.com/user/{github_user}/repos"

# repos = json.loads(requests.get(endpoint).text)
# print(repos)

