"""
Create realistic sample data matching the analysis results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

print("Creating realistic sample dataset...")

# Create data folder if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate dates for 2010-2011 (2 years)
start_date = datetime(2010, 1, 1)
end_date = datetime(2011, 12, 31)
dates = []
current = start_date
while current <= end_date:
    dates.append(current)
    current += timedelta(days=1)

# Create products (matching your analysis)
products = {
    'C003': 'Premium Gift Box Set',
    'B002': 'Decorative Wall Clock', 
    'A001': 'Ceramic Coffee Mug Set',
    'D004': 'Leather Journal',
    'E005': 'Scented Candle Set',
    'F006': 'Wooden Photo Frame',
    'G007': 'Silk Scarf',
    'H008': 'Chocolate Gift Box',
    'I009': 'Silver Earrings',
    'J010': 'Wool Blanket'
}

# Create 800 customers
customers = [f'1{str(i).zfill(5)}' for i in range(1, 801)]

# Countries (UK is dominant like real dataset)
countries = ['United Kingdom'] * 70 + ['France'] * 10 + ['Germany'] * 8 + ['USA'] * 5 + ['Spain'] * 4 + ['Italy'] * 3

print("Generating transactions...")

# Generate transactions with seasonality
data = []
transaction_counter = 50000

for date in dates:
    # More transactions in Nov-Dec (holiday season)
    if date.month in [11, 12]:
        num_transactions = random.randint(80, 120)
    else:
        num_transactions = random.randint(30, 70)
    
    for _ in range(num_transactions):
        # Make C003 the top product (higher probability)
        if random.random() < 0.25:  # 25% chance for C003
            product_code = 'C003'
        else:
            product_code = random.choice(list(products.keys()))
        
        # Higher quantities for C003
        if product_code == 'C003':
            quantity = random.randint(5, 25)
        else:
            quantity = random.randint(1, 10)
        
        price = {
            'C003': 29.99,
            'B002': 24.99,
            'A001': 15.99,
            'D004': 12.99,
            'E005': 18.99,
            'F006': 9.99,
            'G007': 22.99,
            'H008': 14.99,
            'I009': 19.99,
            'J010': 27.99
        }.get(product_code, 15.99)
        
        customer = random.choice(customers)
        country = random.choice(countries)
        
        # Add some weekend effect (less sales on weekends)
        if date.weekday() >= 5:  # Weekend
            if random.random() < 0.7:  # 30% reduction
                continue
        
        data.append({
            'Invoice': f'{transaction_counter}',
            'StockCode': product_code,
            'Description': products[product_code],
            'Quantity': quantity,
            'InvoiceDate': date,
            'Price': price,
            'CustomerID': customer,
            'Country': country
        })
        transaction_counter += 1

# Create DataFrame
df = pd.DataFrame(data)

# Add holiday season boost (already handled by more transactions)
df['TotalAmount'] = df['Quantity'] * df['Price']

print(f"\n✓ Created {len(df)} transactions")
print(f"✓ Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"✓ Products: {df['StockCode'].nunique()}")
print(f"✓ Customers: {df['CustomerID'].nunique()}")

# Show product sales
print("\n PRODUCT SALES:")
product_sales = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
for product, sales in product_sales.head(5).items():
    print(f"   {product}: {sales:,} units")

# Save to Excel
df.to_excel('data/raw/online_retail_II.xlsx', index=False, engine='openpyxl')
print(f"\n✓ Saved to: data/raw/online_retail_II.xlsx")

# Also save a backup as CSV
df.to_csv('data/raw/online_retail_II.csv', index=False)
print(f"✓ Backup saved to: data/raw/online_retail_II.csv")

print(f"\n Data creation complete! {len(df)} transactions ready.")