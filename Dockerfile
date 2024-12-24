# Use a base Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app and test file into the container
COPY main.py .
COPY test_all_companies.py .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run FastAPI with Uvicorn when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
