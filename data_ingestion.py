"""
Simple script to download and read the Online Retail dataset
"""

import pandas as pd
import os
from datetime import datetime

print("=" * 50)
print("STEP 1: DATA INGESTION")
print("=" * 50)

if not os.path.exists('data'):
    os.makedirs('data')
    print("Created 'data' folder")

if not os.path.exists('data/raw'):
    os.makedirs('data/raw')
    print(" Created 'data/raw' folder")

file_path = 'data/raw/online_retail_II.xlsx'

if not os.path.exists(file_path):
    print("\n  Dataset file not found!")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/mathchi/online-retail-ii-data-set-from-ml-repository")
    print("\nSave it as: data/raw/online_retail_II.xlsx")
    print("\nOr you can use this smaller version:")
    
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
        print(f"\nDownloading from {url}...")
        df = pd.read_excel(url, sheet_name='Year 2010-2011', nrows=1000)
        print(" Downloaded sample data (1000 rows)")
    except:
        print(" Could not download automatically")
        exit()
else:
    print(f"\n Found dataset at {file_path}")
    print("Reading data...")
    df = pd.read_excel(file_path)
    print(f" Loaded {len(df)} rows")

print("\n DATASET INFO:")
print(f"   - Total rows: {len(df):,}")
print(f"   - Total columns: {len(df.columns)}")
print(f"   - Columns: {list(df.columns)}")

print("\n FIRST 5 ROWS:")
print(df.head())

info = {
    'source': 'Online Retail II Dataset',
    'rows': len(df),
    'columns': list(df.columns),
    'date_loaded': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

print("\n Data ingestion complete!")