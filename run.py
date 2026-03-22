"""
Main script to run all steps
"""

import os
import sys

print("=" * 60)
print("RETAIL DEMAND FORECASTING PIPELINE")
print("=" * 60)
print("\nThis script will run all steps of the data pipeline:")
print("1. Data Ingestion")
print("2. ETL Pipeline (Data Cleaning)")
print("3. Exploratory Analysis")
print("=" * 60)


run_all = input("\nRun all steps? (y/n): ").lower()

if run_all == 'y':
    print("\n" + "=" * 60)
    print("RUNNING STEP 1: DATA INGESTION")
    print("=" * 60)
    exec(open('data_ingestion.py').read())
    

    print("\n" + "=" * 60)
    print("RUNNING STEP 2: ETL PIPELINE")
    print("=" * 60)
    exec(open('etl_pipeline.py').read())
    
    print("\n" + "=" * 60)
    print("RUNNING STEP 3: EXPLORATORY ANALYSIS")
    print("=" * 60)
    exec(open('analysis.py').read())
    
    print("\n" + "=" * 60)
    print(" ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n Check these folders for outputs:")
    print("   - data/           : Cleaned dataset")
    print("   - reports/        : Analysis results")
    print("   - reports/figures/: Visualizations")
    
else:
    print("\nSkipping pipeline run. Run individual scripts as needed.")