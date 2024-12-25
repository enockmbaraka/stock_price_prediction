# Use a base Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .


RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y build-essential libssl-dev libffi-dev python3-dev



# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app and test file into the container
COPY main.py .

# Copy the models
COPY outputs/stock_price_rnn_google_model.keras .
COPY outputs/stock_price_rnn_meta_model.keras .
COPY outputs/stock_price_lstm_apple_model.keras .
COPY outputs/stock_price_lstm_nvidia_model.keras .

# Copy the scalers
COPY outputs/google_scale.pkl .
COPY outputs/meta_scale.pkl .
COPY outputs/apple_scale.pkl .
COPY outputs/nvidia_scale.pkl .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run FastAPI with Uvicorn when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
