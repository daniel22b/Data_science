import csv
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

df = pd.read_csv("sales_data.csv")

total_sales = df['Sales'].sum()
mean_revenue = df['Revenue'].mean()

print(total_sales, mean_revenue)

plt.figure(figsize=(10,6))

plt.plot(df['Month'], df['Sales'], label = "Sprzedaz" , marker = 'o', linestyle = '-', color = 'blue')
plt.plot(df['Month'], df['Revenue']/100, label = "Przychod" , marker = 'x', linestyle = '-', color = 'red')

sales_by_month = df.groupby('Product')['Sales'].sum()
print(sales_by_month)

sorted_by_revenue = df.sort_values(by='Revenue', ascending=False)
print(sorted_by_revenue)

product_a_sales = df[df['Product'] == 'Product A']
product_b_sales = df[df['Product'] == 'Product B']


plt.plot(product_a_sales['Month'], product_a_sales['Sales'], color ='r')
plt.plot(product_b_sales['Month'], product_b_sales['Sales'], color ='b')
plt.show()



# plt.title('Sprzedaż i przychód w czasie')
# plt.xlabel('Miesiące')
# plt.ylabel('Wartości')
# plt.legend()
# plt.xticks(rotation=45)  # Obróć etykiety na osi X dla lepszej czytelności
# plt.tight_layout()  # Zapewnia, że etykiety będą dobrze widoczne
# plt.show()


# total_sales = []
# mean_revenue = []
# with open("sales_data.csv",'r') as f:
#     file = csv.DictReader(f, delimiter=',')
    

#     for line in file:
#         month = line["Month"]
#         product = line["Product"]
#         sales = int(line["Sales"])
#         revenue = int(line["Revenue"])

        
#         total_sales.append(sales)
#         mean_revenue.append(revenue)

#         #sum of sold products
# sum_of_sales = sum(total_sales)
#         #mean of ravenue
# mean = np.mean(mean_revenue)


    

