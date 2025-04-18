import csv
import sys, re

def process(symbol, date, closing_price):
    print(f"ID: {symbol}, Date: {date}, Closing price: {closing_price}")

with open("comma_delimited_stock_prices.csv", "r") as f:
    tab_reader = csv.reader(f, delimiter=',')
    for row in tab_reader:
        date = row[0]
        symbol = row[1]
        closing_price = row[2]
        process(date, symbol, closing_price)

with open("comma_delimited_stock_prices.csv", "r") as f:
    colon_reder = csv.DictReader(f, delimiter=':')
    for dict_row in colon_reder:
        symbol = dict_row["symbol"]
        date = dict_row["date"]
        closing_price = dict_row["closing_price"]
        process(symbol,date,closing_price)
     
today_prcies  = {'APPL':29.34, 'NVD': 45.23, 'CDP': 455.21}

with open("comma_delimited_stock_prices.csv", "w") as f:
    csv_writer = csv.writer(f, delimiter=':')

    for stock, price in today_prcies.items():
        csv_writer.writerow([stock, price] )

with open('stocks.csv', 'r') as f:
    x = csv.DictReader(f, delimiter=',')
    for row in x:
        symbol = row["Symbol"]
        date = row["Date"]
        high = row["High"]
        low = row["Low"]
        print(f"symbol: {symbol} date:{date} high: {high} low: {low}")



