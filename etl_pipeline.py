"""
ETL Pipeline - Clean and transform the data
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("\n" + "=" * 50)
print("STEP 2: ETL PIPELINE")
print("=" * 50)

print("\n Reading data...")
try:
    df = pd.read_excel('data/raw/online_retail_II.xlsx', sheet_name='Year 2010-2011')
    print(f" Loaded {len(df):,} rows")
except:
    print("  Creating sample data for testing...")
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Invoice': [f'INV{i}' for i in range(1000)],
        'StockCode': np.random.choice(['A001', 'B002', 'C003'], 1000),
        'Description': ['Product A', 'Product B', 'Product C'] * 333 + ['Product A'],
        'Quantity': np.random.randint(1, 20, 1000),
        'InvoiceDate': np.random.choice(dates, 1000),
        'Price': np.random.uniform(5, 50, 1000),
        'CustomerID': np.random.choice(['C001', 'C002', 'C003', None], 1000),
        'Country': np.random.choice(['UK', 'USA', 'France', 'Germany'], 1000)
    })
    print(f"Created sample data with {len(df)} rows")

print(f"\n BEFORE CLEANING:")
print(f"   - Rows: {len(df)}")
print(f"   - Missing values: {df.isnull().sum().sum()}")



print("\n 1. HANDLING MISSING VALUES")
initial_rows = len(df)

missing_customer = df['CustomerID'].isnull().sum()
df = df[df['CustomerID'].notna()]
print(f"   - Removed {missing_customer} rows with missing CustomerID")




if 'Description' in df.columns:
    missing_desc = df['Description'].isnull().sum()
    df = df[df['Description'].notna()]
    print(f"   - Removed {missing_desc} rows with missing Description")

print("\n 2. REMOVING INVALID DATA")
initial = len(df)

df = df[df['Quantity'] > 0]
print(f"   - Removed {initial - len(df)} rows with negative/zero quantity")


initial = len(df)
df = df[df['Price'] > 0]
print(f"   - Removed {initial - len(df)} rows with negative/zero price")



if 'Invoice' in df.columns:
    df = df[~df['Invoice'].astype(str).str.contains('C', na=False)]
    print(f"   - Removed cancelled invoices")

print("\n 3. ADDING DERIVED COLUMNS")



df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['TotalAmount'] = df['Quantity'] * df['Price']
print(f"   - Added 'TotalAmount' column")

df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
print(f"   - Added Year, Month, Day, DayOfWeek columns")

df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])
df['IsHolidaySeason'] = df['Month'].isin([11, 12])
print(f"   - Added IsWeekend, IsHolidaySeason flags")

print(f"\n AFTER CLEANING:")
print(f"   - Rows: {len(df)}")
print(f"   - Columns: {len(df.columns)}")
print(f"   - Missing values: {df.isnull().sum().sum()}")

print("\n Saving cleaned data...")
df.to_parquet('data/cleaned_data.parquet')
print(f"Saved to: data/cleaned_data.parquet")

stats = {
    'original_rows': initial_rows,
    'cleaned_rows': len(df),
    'percent_retained': (len(df) / initial_rows) * 100,
    'cleaning_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

print(f"\n CLEANING STATISTICS:")
print(f"   - Original rows: {stats['original_rows']:,}")
print(f"   - Cleaned rows: {stats['cleaned_rows']:,}")
print(f"   - Retained: {stats['percent_retained']:.1f}%")

print("\n ETL pipeline complete!")