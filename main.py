from typing import List
from pydantic import BaseModel, Field, ValidationError
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
import os

# Custom metric registration
@tf.keras.utils.register_keras_serializable(package="Custom", name="symmetric_mean_absolute_percentage_error")
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    numerator = tf.abs(y_true - y_pred)
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2
    smape = tf.reduce_mean(numerator / denominator) * 100
    return smape

# FastAPI initialization
app = FastAPI()

# Define input data model with validation
class StockData(BaseModel):
    google: List[float] = Field(..., description="Google stock time series with 60 values")
    meta: List[float] = Field(..., description="Meta stock time series with 60 values")
    apple: List[float] = Field(..., description="Apple stock time series with 60 values")
    nvidia: List[float] = Field(..., description="NVIDIA stock time series with 60 values")

    @staticmethod
    def validate_time_series_length(series, name):
        if len(series) != 60:
            raise ValueError(f"{name} time series must have exactly 60 elements.")
        return series

    @classmethod
    def validate(cls, data):
        for company in ["google", "meta", "apple", "nvidia"]:
            cls.validate_time_series_length(data[company], company)
        return data

# Model and preprocessor loader
class ModelLoader:
    def __init__(self, model_paths, preprocessor_paths):
        self.model_paths = model_paths
        self.preprocessor_paths = preprocessor_paths
        self.models = {}
        self.preprocessors = {}

    def load_models(self):
        for name, path in self.model_paths.items():
            if os.path.exists(path):
                self.models[name] = load_model(path)
                print(f"Loaded model: {name} from {path}")
            else:
                raise FileNotFoundError(f"Model file not found: {path}")

    def load_preprocessors(self):
        for name, path in self.preprocessor_paths.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.preprocessors[name] = pickle.load(f)
                print(f"Loaded preprocessor: {name} from {path}")
            else:
                raise FileNotFoundError(f"Preprocessor file not found: {path}")

    def get_model(self, name):
        return self.models.get(name)

    def get_preprocessor(self, name):
        return self.preprocessors.get(name)

# Paths for models and preprocessors
model_paths = {
    "google": "outputs/stock_price_rnn_google_model.keras",
    "meta": "outputs/stock_price_rnn_meta_model.keras",
    "apple": "outputs/stock_price_lstm_apple_model.keras",
    "nvidia": "outputs/stock_price_lstm_nvidia_model.keras"
}

preprocessor_paths = {
    "google": "outputs/google_scale.pkl",
    "meta": "outputs/meta_scale.pkl",
    "apple": "outputs/apple_scale.pkl",
    "nvidia": "outputs/nvidia_scale.pkl"
}

# Instantiate loader and load resources
model_loader = ModelLoader(model_paths, preprocessor_paths)
model_loader.load_models()
model_loader.load_preprocessors()

# Prediction endpoint
@app.post("/predict")
def predict(data: StockData):
    try:
        StockData.validate(data.dict())  # Validate time series lengths
        results = {}
        for company in ["google", "meta", "apple", "nvidia"]:
            input_series = np.array(getattr(data, company)).reshape(-1, 1)
            preprocessor = model_loader.get_preprocessor(company)
            model = model_loader.get_model(company)

            if not preprocessor or not model:
                raise ValueError(f"Model or preprocessor for {company} not found.")

            scaled_series = preprocessor.transform(input_series).reshape(1, -1, 1)
            prediction = model.predict(scaled_series)
            predicted_price = preprocessor.inverse_transform(prediction)[0][0]
            results[f"predicted_close_price_{company}"] = float(predicted_price)

        return results
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
