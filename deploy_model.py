"""
Deploy Model to Azure
Phase 2 - Model deployment and validation
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("=" * 60)
print("MODEL DEPLOYMENT")
print("=" * 60)

# Load model and artifacts
print("\n Loading model...")
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_cols = pd.read_csv('models/feature_columns.csv')['features'].tolist()

print(f" Model loaded")
print(f" Features: {len(feature_cols)}")

# 
# SECTION 1: BATCH PREDICTION FUNCTION
# 
def predict_demand(input_data):
    """
    Predict demand for given input features
    
    Parameters:
    input_data: DataFrame with required features
    
    Returns:
    predictions: array of predictions
    """
    # Ensure features are in correct order
    X = input_data[feature_cols]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    
    return predictions

# 
# SECTION 2: TEST WITH SAMPLE DATA
# 
print("\n Testing with sample data...")

# Create sample input (using last day from training)
sample_input = pd.DataFrame({
    'day_of_week': [1],
    'month': [3],
    'quarter': [1],
    'is_weekend': [0],
    'is_holiday_season': [0],
    'lag_1_quantity': [50],
    'lag_2_quantity': [45],
    'lag_3_quantity': [55],
    'lag_7_quantity': [48],
    'rolling_mean_7_quantity': [52],
    'rolling_mean_14_quantity': [49],
    'rolling_mean_30_quantity': [51],
    'rolling_std_7_quantity': [8],
    'rolling_std_14_quantity': [10],
    'rolling_std_30_quantity': [12]
})

try:
    prediction = predict_demand(sample_input)
    print(f" Sample prediction: {prediction[0]:.2f} units")
except Exception as e:
    print(f" Prediction failed: {e}")

# 
# SECTION 3: CREATE BATCH PREDICTION SCRIPT
# 
print("\n Creating batch prediction script...")

batch_script = """
import pandas as pd
import joblib
import sys
import os

def predict_demand_batch(input_file, output_file):
    # Load model
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_cols = pd.read_csv('models/feature_columns.csv')['features'].tolist()
    
    # Load input data
    df = pd.read_csv(input_file)
    
    # Prepare features
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    
    # Save results
    df['predicted_demand'] = predictions
    df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")
    return predictions

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_predict.py input.csv output.csv")
        sys.exit(1)
    
    predict_demand_batch(sys.argv[1], sys.argv[2])
"""

with open('models/batch_predict.py', 'w') as f:
    f.write(batch_script)

print(" Created batch_predict.py")

# SECTION 4: DEPLOYMENT VALIDATION
print("\n Running deployment validation...")

# Load test data
df = pd.read_parquet('data/cleaned_data.parquet')

# Create daily features (same as training)
daily_df = df.groupby(df['InvoiceDate'].dt.date).agg({
    'Quantity': 'sum',
    'TotalAmount': 'sum',
    'Price': 'mean',
    'CustomerID': 'nunique'
}).reset_index()

daily_df.columns = ['date', 'total_quantity', 'total_revenue', 'avg_price', 'unique_customers']
daily_df['date'] = pd.to_datetime(daily_df['date'])

# Create features
daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
daily_df['month'] = daily_df['date'].dt.month
daily_df['quarter'] = daily_df['date'].dt.quarter
daily_df['is_weekend'] = (daily_df['day_of_week'] >= 5).astype(int)
daily_df['is_holiday_season'] = (daily_df['month'].isin([11, 12])).astype(int)

# Create lag features
for lag in [1, 2, 3, 7]:
    daily_df[f'lag_{lag}_quantity'] = daily_df['total_quantity'].shift(lag)

for window in [7, 14, 30]:
    daily_df[f'rolling_mean_{window}_quantity'] = daily_df['total_quantity'].rolling(window, min_periods=1).mean()
    daily_df[f'rolling_std_{window}_quantity'] = daily_df['total_quantity'].rolling(window, min_periods=1).std()

daily_df = daily_df.dropna()

# Validate predictions
X_validation = daily_df[feature_cols]
X_scaled = scaler.transform(X_validation)
validation_predictions = model.predict(X_scaled)

validation_metrics = {
    'predictions_count': len(validation_predictions),
    'min_prediction': validation_predictions.min(),
    'max_prediction': validation_predictions.max(),
    'mean_prediction': validation_predictions.mean(),
    'std_prediction': validation_predictions.std()
}

print(f"\n Validation Metrics:")
print(f"   - Predictions: {validation_metrics['predictions_count']}")
print(f"   - Range: {validation_metrics['min_prediction']:.2f} to {validation_metrics['max_prediction']:.2f}")
print(f"   - Mean: {validation_metrics['mean_prediction']:.2f}")
print(f"   - Std Dev: {validation_metrics['std_prediction']:.2f}")

# 
# SECTION 5: DEPLOYMENT CONFIGURATION
print("\n Creating deployment configuration...")

deploy_config = f"""
# Model Deployment Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- **Model File**: models/best_model.pkl
- **Scaler File**: models/scaler.pkl
- **Features File**: models/feature_columns.csv
- **Model Type**: {type(model).__name__}

## Deployment Mode: Batch
- **Input Format**: CSV with required features
- **Output Format**: CSV with predictions added
- **Batch Script**: models/batch_predict.py

## Feature Requirements
Required features for prediction:
{chr(10).join([f'- {col}' for col in feature_cols])}

## Validation Results
- **Validation Predictions**: {validation_metrics['predictions_count']}
- **Prediction Range**: [{validation_metrics['min_prediction']:.2f}, {validation_metrics['max_prediction']:.2f}]
- **Average Demand**: {validation_metrics['mean_prediction']:.2f}

## Azure Deployment
- **Storage**: Azure Blob Storage
- **Compute**: Azure Functions (serverless)
- **Schedule**: Daily batch predictions
"""

with open('models/deployment_config.md', 'w') as f:
    f.write(deploy_config)

print("Created deployment_config.md")

print("\n" + "=" * 60)
print(" DEPLOYMENT READY!")
print("=" * 60)
print("\n Deployment artifacts:")
print("   - models/best_model.pkl")
print("   - models/scaler.pkl")
print("   - models/batch_predict.py")
print("   - models/deployment_config.md")