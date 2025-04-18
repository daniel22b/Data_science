import datetime
import pandas as pd
from collections import namedtuple

Stock_price = namedtuple('self', ['symbol', 'date', 'closing_price'])
price = Stock_price('MSFT', datetime.date(2013,12,30), 105.34)
print(price.closing_price)

stock_price2 = {'cloesig price': [102.06],
               'date': [datetime.date(2014, 8, 29)],
               'symbol':['AAPL']}

df = pd.DataFrame(stock_price2)


#PROSTA KLASA
# class kalkulator:
#     def __init__(self, a,b):
#         self.a = a
#         self.b = b
    
#     def add(self, c):
#         return self.a + self.b +c
    
#     def dev(self):
#         return self.a - self.b

# x = kalkulator(2,3)

# print(x.add(2))