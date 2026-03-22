
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
