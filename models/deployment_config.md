
# Model Deployment Configuration
# Generated: 2026-03-22 17:10:07

## Model Information
- **Model File**: models/best_model.pkl
- **Scaler File**: models/scaler.pkl
- **Features File**: models/feature_columns.csv
- **Model Type**: RandomForestRegressor

## Deployment Mode: Batch
- **Input Format**: CSV with required features
- **Output Format**: CSV with predictions added
- **Batch Script**: models/batch_predict.py

## Feature Requirements
Required features for prediction:
- day_of_week
- month
- quarter
- is_weekend
- is_holiday_season
- lag_1_quantity
- lag_2_quantity
- lag_3_quantity
- lag_7_quantity
- rolling_mean_7_quantity
- rolling_mean_14_quantity
- rolling_mean_30_quantity
- rolling_std_7_quantity
- rolling_std_14_quantity
- rolling_std_30_quantity

## Validation Results
- **Validation Predictions**: 92
- **Prediction Range**: [32.77, 159.29]
- **Average Demand**: 79.31

## Azure Deployment
- **Storage**: Azure Blob Storage
- **Compute**: Azure Functions (serverless)
- **Schedule**: Daily batch predictions
