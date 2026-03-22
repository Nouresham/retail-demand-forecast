## Phase 2: Model Development Results

### Best Model: Random Forest
- **Test RMSE**: 29.91 units
- **Test R²**: -0.2445
- **Features Used**: 15 (temporal, lags, rolling statistics)

### Model Performance Comparison

| Model | Test RMSE | Improvement vs Baseline |
|-------|-----------|------------------------|
| Baseline (Moving Avg) | 29.71 | - |
| **Random Forest** | **29.91** | **-0.7%** |
| Gradient Boosting | 32.99 | -11.0% |
| Ridge Regression | 43.34 | -45.9% |
| Linear Regression | 53.85 | -81.2% |

### Validation Strategy
- **Split**: Time-based (80% train / 20% test)
- **Metrics**: RMSE, MAE, R²
- **No Data Leakage**: Features use only past information

### Deployment
- **Mode**: Batch prediction
- **Model File**: `models/best_model.pkl`
- **Prediction Script**: `models/batch_predict.py`
- **Input**: CSV with 15 features
- **Output**: CSV with predictions

### How to Use the Model

```bash
# Make predictions
python models/batch_predict.py input.csv output.csv