from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Define the input data model for stock price prediction
class StockData(BaseModel):
    time_series: list[float]  # Input list of past stock prices for prediction

# Load the .keras model and preprocessing logic (once when the app starts)
model = load_model("outputs/stock_price_rnn_model.keras")  # Load the .keras model
with open("outputs/google_scale.pkl", 'rb') as f_in:
    preprocessor = pickle.load(f_in)  # Load the preprocessing object

# Define an endpoint for prediction
@app.post("/predict")
def predict(data: StockData):
    # Convert input data to a NumPy array
    input_series = np.array(data.time_series).reshape(1, -1)  # Reshape to match model's expected input

    # Preprocess the input series
    preprocessed_series = preprocessor.transform(input_series)  # Apply preprocessing logic

    # Make prediction using the loaded .keras model
    prediction = model.predict(preprocessed_series)

    # If the model predicts a single future value, return it
    return {"predicted_close_price": float(prediction[0][0])}

# If running with Uvicorn (use this in the terminal):
# uvicorn app:app --reload
