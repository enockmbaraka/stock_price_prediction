# Use a base Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update -y && apt-get install -y gcc  python3-dev


# Upgrade pip to the latest version
RUN pip install --upgrade pip setuptools wheel --timeout 1000

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies without using the cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application file
COPY main.py .

# Copy the RNN and LSTM model files
COPY outputs/stock_price_rnn_google_model.keras .
COPY outputs/stock_price_rnn_meta_model.keras .
COPY outputs/stock_price_lstm_apple_model.keras .
COPY outputs/stock_price_lstm_nvidia_model.keras .

# Copy the scaler files
COPY outputs/google_scale.pkl .
COPY outputs/meta_scale.pkl .
COPY outputs/apple_scale.pkl .
COPY outputs/nvidia_scale.pkl .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
