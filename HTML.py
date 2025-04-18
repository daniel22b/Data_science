from bs4 import BeautifulSoup
import requests
from collections.abc import Callable 

url = ("https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html")

html = requests.get(url).text
soup = BeautifulSoup(html, 'html.parser')

print(soup.prettify())

first_paragraph = soup.find('p')
print(first_paragraph)

first_paragraph_text = soup.p.text
print(first_paragraph_text)

first_paragraph_word = soup.p.text.split()
print(first_paragraph_word)

first_paragraph_id = soup.p['id']
print(first_paragraph_id)

first_paragraph_id2 = soup.p.get('id')
print(first_paragraph_id2)

all_paragraphs = soup.find_all('p')
print(all_paragraphs)

paragraphs_with_ids = [p for p in soup('p') if p.get('id')]
print(paragraphs_with_ids)

important_paragraphs = soup('p', {'class': 'important '})
print(important_paragraphs)

important_paragraphs2 = soup('p', 'ipmortant ')
print(important_paragraphs2)

important_paragraphs3 = soup(p for p in soup('p')
                             if 'ipmportant' in p.get('class', []))
print(important_paragraphs3)

spans_inside_divs = [span for div in soup('div') for span in div('span')]
print(spans_inside_divs)
