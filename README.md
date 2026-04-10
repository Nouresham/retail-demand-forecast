# Online Retail Demand Forecasting System

**Student:** Nouresham Katrmiz  
**Course:** DSAI3202  
**Date:** March 22, 2026

## Project Overview
AI-driven demand forecasting system for small online retail shops to optimize inventory management.

## Phase 1: Data Pipeline

### Data Ingestion
- **Source**: Online Retail II Dataset (Kaggle/UCI)
- **Format**: Excel (.xlsx)
- **Storage**: `data/raw/online_retail_II.xlsx`

### ETL Pipeline Results
- **Cleaned Records**: 730 rows
- **Total Revenue**: £189,199.25
- **Unique Products**: 3

### Visualizations Generated
- daily_sales.png, monthly_sales.png, top_products.png, daily_pattern.png, correlation.png

## Phase 2: Model Development

### Best Model: Random Forest
- **Test RMSE**: 29.91 units
- **Test R²**: -0.2445

### Model Performance
| Model | Test RMSE |
|-------|-----------|
| Baseline | 29.71 |
| Random Forest | 29.91 |
| Gradient Boosting | 32.99 |

### How to Use
```bash
python models/batch_predict.py input.csv output.csv
