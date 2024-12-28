# Use a base Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install essential tools and Python development headers
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel to the latest version
RUN pip install --upgrade pip setuptools wheel --timeout 1000

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies without using the cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY main.py .

# Ensure the outputs directory exists and copy all required model and scaler files
RUN mkdir -p outputs
COPY outputs/stock_price_rnn_google_model.keras outputs/
COPY outputs/stock_price_rnn_meta_model.keras outputs/
COPY outputs/stock_price_lstm_apple_model.keras outputs/
COPY outputs/stock_price_lstm_nvidia_model.keras outputs/
COPY outputs/google_scale.pkl outputs/
COPY outputs/meta_scale.pkl outputs/
COPY outputs/apple_scale.pkl outputs/
COPY outputs/nvidia_scale.pkl outputs/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
