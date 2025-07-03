import pandas as pd
from prophet import Prophet
import joblib
import os
from datetime import datetime

def train_forecast_models():
    """Train forecast models for each city"""
    print("Starting model training...")
    df = pd.read_csv('weather_data/training_dataset.csv')
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    models = {}
    for city in df['city'].unique():
        city_df = df[df['city'] == city]
        city_df = city_df.rename(columns={'timestamp': 'ds', 'temperature': 'y'})
        
        # Initialize and fit model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        model.fit(city_df)
        models[city] = model
        print(f"✅ Trained model for {city}")
    
    # Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = f"models/weather_models_{timestamp}.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(models, model_path)
    
    # Update latest model reference
    with open("models/latest_model.txt", "w") as f:
        f.write(model_path)
    
    print(f"Saved models to {model_path}")
    return model_path

if __name__ == "__main__":
    train_forecast_models()