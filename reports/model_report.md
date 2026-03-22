
# Model Development Report
**Date:** 2026-03-22 17:10:02

## Data Summary
- **Total Records**: 740 transactions
- **Daily Records**: 69 days
- **Train Period**: 2023-01-31 00:00:00 to 2023-03-27 00:00:00
- **Test Period**: 2023-03-28 00:00:00 to 2023-04-10 00:00:00
- **Train/Test Split**: 55 / 14 rows

## Features Created
- **Temporal**: day_of_week, month, quarter, is_weekend, is_holiday_season
- **Lag Features**: 1, 2, 3, 7, 14, 30 days
- **Rolling Statistics**: 7, 14, 30 day means and standard deviations

## Model Performance

### Baseline (7-day Moving Average)
- **RMSE**: 29.71
- **MAE**: 24.66

### Machine Learning Models
| Model | Train RMSE | Test RMSE | Test MAE | Test R² |
|-------|------------|-----------|----------|---------|
| Linear Regression | 22.37 | 53.85 | 37.08 | -3.0345 |
| Ridge Regression | 22.52 | 43.34 | 31.61 | -1.6143 |
| Random Forest | 11.50 | 29.91 | 23.67 | -0.2445 |
| Gradient Boosting | 1.69 | 32.99 | 25.05 | -0.5142 |

## Best Model: Random Forest
- **Test RMSE**: 29.91
- **Improvement over Baseline**: -0.6%

## Validation Strategy
- **Split Method**: Time-based (respects temporal order)
- **Evaluation Metrics**: RMSE, MAE, R²
- **No Data Leakage**: Features only use past information

## Model Registration
- **Model ID**: 20260322_171002
- **Training Data Version**: Phase 2 cleaned data
- **Features**: 15 features
- **File Location**: `models/best_model.pkl`

## Limitations
- Limited historical data (69 days)
- Only 3 products in dataset
- No external factors (weather, holidays, promotions)
