# Use a base Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies and Rust
RUN apt-get update && apt-get install -y curl build-essential libssl-dev libffi-dev python3-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . "/root/.cargo/env"

# Add Rust binary to PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure that `project.version` is correctly set in pyproject.toml to avoid maturin issues
#COPY pyproject.toml . 
# Copy the FastAPI app
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
