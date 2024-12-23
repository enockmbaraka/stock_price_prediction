import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

@tf.keras.utils.register_keras_serializable(package="Custom", name="symmetric_mean_absolute_percentage_error")
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Computes Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        SMAPE as a TensorFlow tensor.
    """
    numerator = tf.abs(y_true - y_pred)
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2
    smape = tf.reduce_mean(numerator / denominator) * 100
    return smape


# Initialize FastAPI app
app = FastAPI()

# Define the input data model for stock price prediction
class StockData(BaseModel):
    time_series: conlist(float)  # Enforce list of floats with at least one item

# Load the .keras model and preprocessing logic (once when the app starts)
model = load_model("outputs/stock_price_rnn_google_model.keras")  # Load the .keras model
with open("outputs/google_scale.pkl", 'rb') as f_in:
    preprocessor = pickle.load(f_in)  # Load the preprocessing object

# Define an endpoint for prediction
@app.post("/predict")
def predict(data: StockData):
    try:
        # Convert input data to a NumPy array
        input_series = np.array(data.time_series).reshape(-1, 1)  # Reshape to (60, 1) for per-element scaling

        # Initialize a list to store scaled values
        scaled_values = []

        # Apply the scaler to each element individually
        for value in input_series:
            # Each value is a 2D array, as MinMaxScaler expects 2D data
            scaled_value = preprocessor.transform(value.reshape(1, -1))  # Reshape for single element
            scaled_values.append(scaled_value[0][0])  # Get the scaled value and append to list

        # Convert the list back to a NumPy array
        scaled_values = np.array(scaled_values).reshape(1, -1)  # Reshape back to (1, 60)

        # Reshape the scaled data back to (1, 60, 1) for model input
        scaled_values = scaled_values.reshape((scaled_values.shape[0], scaled_values.shape[1], 1))

        # Make prediction using the loaded .keras model
        prediction = model.predict(scaled_values)

        # Return the predicted close price
        return {"predicted_close_price": float(preprocessor.inverse_transform(prediction[0][0].reshape(1, -1)))}

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

# If running with Uvicorn (use this in the terminal):
# uvicorn app:app --reload
