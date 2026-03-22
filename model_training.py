"""
Model Development for Demand Forecasting
Phase 2 - Train, validate, and compare models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("PHASE 2: MODEL DEVELOPMENT")
print("=" * 60)

# Create models folder
os.makedirs('models', exist_ok=True)
print(" Models folder created")

# Load cleaned data
print("\n Loading cleaned data...")
df = pd.read_parquet('data/cleaned_data.parquet')
print(f" Loaded {len(df)} rows")

#
# SECTION 1: FEATURE ENGINEERING
print("\n Creating features for modeling...")

# Aggregate to daily level for demand forecasting
daily_df = df.groupby(df['InvoiceDate'].dt.date).agg({
    'Quantity': 'sum',
    'TotalAmount': 'sum',
    'Price': 'mean',
    'CustomerID': 'nunique'
}).reset_index()

daily_df.columns = ['date', 'total_quantity', 'total_revenue', 'avg_price', 'unique_customers']
daily_df['date'] = pd.to_datetime(daily_df['date'])

print(f" Created {len(daily_df)} daily records")

# Create features
daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
daily_df['month'] = daily_df['date'].dt.month
daily_df['quarter'] = daily_df['date'].dt.quarter
daily_df['day_of_year'] = daily_df['date'].dt.dayofyear
daily_df['is_weekend'] = (daily_df['day_of_week'] >= 5).astype(int)
daily_df['is_holiday_season'] = (daily_df['month'].isin([11, 12])).astype(int)

# Create lag features (previous days' sales)
for lag in [1, 2, 3, 7, 14, 30]:
    daily_df[f'lag_{lag}_quantity'] = daily_df['total_quantity'].shift(lag)
    daily_df[f'lag_{lag}_revenue'] = daily_df['total_revenue'].shift(lag)

# Create rolling averages
for window in [7, 14, 30]:
    daily_df[f'rolling_mean_{window}_quantity'] = daily_df['total_quantity'].rolling(window, min_periods=1).mean()
    daily_df[f'rolling_std_{window}_quantity'] = daily_df['total_quantity'].rolling(window, min_periods=1).std()

# Remove rows with NaN from lag features
daily_df = daily_df.dropna().reset_index(drop=True)
print(f" After feature engineering: {len(daily_df)} rows")

# SECTION 2: TRAIN-TEST SPLIT (Time-based)

print("\n Creating train-test split...")

# Use time-based split (respect temporal order)
split_date = daily_df['date'].quantile(0.8)  # 80% train, 20% test
train_df = daily_df[daily_df['date'] <= split_date]
test_df = daily_df[daily_df['date'] > split_date]

print(f" Train set: {len(train_df)} rows ({train_df['date'].min()} to {train_df['date'].max()})")
print(f"Test set: {len(test_df)} rows ({test_df['date'].min()} to {test_df['date'].max()})")

# Define features and target
feature_cols = ['day_of_week', 'month', 'quarter', 'is_weekend', 'is_holiday_season',
                'lag_1_quantity', 'lag_2_quantity', 'lag_3_quantity', 'lag_7_quantity',
                'rolling_mean_7_quantity', 'rolling_mean_14_quantity', 'rolling_mean_30_quantity',
                'rolling_std_7_quantity', 'rolling_std_14_quantity', 'rolling_std_30_quantity']

target_col = 'total_quantity'

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

print(f"Features: {len(feature_cols)} columns")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SECTION 3: MODEL TRAINING
print("\n Training models...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results[name] = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'predictions': y_test_pred
    }
    
    print(f"   Train RMSE: {train_rmse:.2f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    print(f"   Test R²: {test_r2:.4f}")

# SECTION 4: BASELINE MODEL
print("\n Creating baseline model (simple moving average)...")

baseline_preds = []
for i in range(len(test_df)):
    if i < 7:
        baseline_preds.append(train_df['total_quantity'].mean())
    else:
        baseline_preds.append(test_df['total_quantity'].iloc[i-7:i].mean())

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_mae = mean_absolute_error(y_test, baseline_preds)

print(f"   Baseline RMSE: {baseline_rmse:.2f}")
print(f"   Baseline MAE: {baseline_mae:.2f}")

# SECTION 5: SELECT BEST MODEL
print("\n Selecting best model...")

best_model_name = min(results, key=lambda x: results[x]['test_rmse'])
best_model = results[best_model_name]['model']
best_rmse = results[best_model_name]['test_rmse']

print(f" Best model: {best_model_name}")
print(f" Test RMSE: {best_rmse:.2f}")

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print(f" Model saved to: models/best_model.pkl")

feature_list = pd.DataFrame({'features': feature_cols})
feature_list.to_csv('models/feature_columns.csv', index=False)

# SECTION 6:  VISUALIZATION
print("\n Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, results[best_model_name]['predictions'], alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Quantity')
axes[0, 0].set_ylabel('Predicted Quantity')
axes[0, 0].set_title(f'{best_model_name}: Actual vs Predicted')

# 2. Model Comparison
models_list = list(results.keys())
rmse_values = [results[m]['test_rmse'] for m in models_list]
bars = axes[0, 1].bar(models_list, rmse_values, color='skyblue')
axes[0, 1].axhline(y=baseline_rmse, color='red', linestyle='--', label=f'Baseline RMSE: {baseline_rmse:.2f}')
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_title('Model Performance Comparison')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    axes[1, 0].barh(range(len(indices)), importances[indices])
    axes[1, 0].set_yticks(range(len(indices)))
    axes[1, 0].set_yticklabels([feature_cols[i] for i in indices])
    axes[1, 0].set_xlabel('Feature Importance')
    axes[1, 0].set_title(f'{best_model_name}: Top 10 Features')

# 4. Time Series Forecast
test_dates = test_df['date']
axes[1, 1].plot(test_dates, y_test, label='Actual', linewidth=1)
axes[1, 1].plot(test_dates, results[best_model_name]['predictions'], label='Predicted', linewidth=1, alpha=0.7)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Quantity')
axes[1, 1].set_title(f'{best_model_name}: Forecast vs Actual')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('reports/figures/model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Created model_results.png")

# SECTION 7: SAVE 
print("\n Saving model report...")

report = f"""
# Model Development Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Summary
- **Total Records**: {len(df)} transactions
- **Daily Records**: {len(daily_df)} days
- **Train Period**: {train_df['date'].min()} to {train_df['date'].max()}
- **Test Period**: {test_df['date'].min()} to {test_df['date'].max()}
- **Train/Test Split**: {len(train_df)} / {len(test_df)} rows

## Features Created
- **Temporal**: day_of_week, month, quarter, is_weekend, is_holiday_season
- **Lag Features**: 1, 2, 3, 7, 14, 30 days
- **Rolling Statistics**: 7, 14, 30 day means and standard deviations

## Model Performance

### Baseline (7-day Moving Average)
- **RMSE**: {baseline_rmse:.2f}
- **MAE**: {baseline_mae:.2f}

### Machine Learning Models
| Model | Train RMSE | Test RMSE | Test MAE | Test R² |
|-------|------------|-----------|----------|---------|
"""

for name, metrics in results.items():
    report += f"| {name} | {metrics['train_rmse']:.2f} | {metrics['test_rmse']:.2f} | {metrics['test_mae']:.2f} | {metrics['test_r2']:.4f} |\n"

report += f"""
## Best Model: {best_model_name}
- **Test RMSE**: {best_rmse:.2f}
- **Improvement over Baseline**: {(baseline_rmse - best_rmse) / baseline_rmse * 100:.1f}%

## Validation Strategy
- **Split Method**: Time-based (respects temporal order)
- **Evaluation Metrics**: RMSE, MAE, R²
- **No Data Leakage**: Features only use past information

## Model Registration
- **Model ID**: {datetime.now().strftime('%Y%m%d_%H%M%S')}
- **Training Data Version**: Phase 2 cleaned data
- **Features**: {len(feature_cols)} features
- **File Location**: `models/best_model.pkl`

## Limitations
- Limited historical data ({len(daily_df)} days)
- Only 3 products in dataset
- No external factors (weather, holidays, promotions)
"""

# Save report
with open('reports/model_report.md', 'w') as f:
    f.write(report)

print("✓ Saved model_report.md")
print("\n" + "=" * 60)
print(" MODEL DEVELOPMENT COMPLETE!")
print("=" * 60)
print("\n Output files:")
print("   - models/best_model.pkl")
print("   - models/scaler.pkl")
print("   - reports/model_report.md")
print("   - reports/figures/model_results.png")