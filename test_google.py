import requests

# Define the FastAPI endpoint URL
url = "http://127.0.0.1:8000/predict"

# Example time-series data (flat list)
test_time_series = [
    141.47000122, 143.47999573, 146.38000488, 145.99000549, 147.03999329,
    148.69999695, 151.86999512, 152.19000244, 153.50999451, 151.46000671,
    140.1000061, 141.16000366, 142.38000488, 143.67999268, 144.1000061,
    145.53999329, 145.91000366, 149.0, 147.52999878, 145.13999939,
    145.94000244, 142.77000427, 140.52000427, 141.11999512, 142.55000305,
    144.08999634, 143.96000671, 137.57000732, 138.88000488, 136.38000488,
    138.46000671, 137.13999939, 133.3500061, 132.66999817, 131.3999939,
    134.38000488, 135.41000366, 137.66999817, 138.5, 139.78999329,
    143.1000061, 141.17999268, 147.67999268, 147.02999878, 148.74000549,
    147.6000061, 150.77000427, 150.07000732, 150.66999817, 150.86999512,
    150.92999268, 155.49000549, 154.55999756, 154.91999817, 150.52999878,
    152.5, 154.8500061, 156.6000061, 156.13999939, 159.41000366
]


# Prepare the input for the API
test_data = {
    "time_series": test_time_series  # Flat list of floats
}

# Send POST request to FastAPI
response = requests.post(url, json=test_data)

# Check the response
if response.status_code == 200:
    prediction_result = response.json()
    print("Prediction result:", prediction_result)
else:
    print(f"Request failed with status code {response.status_code}")
    print("Error:", response.text)
