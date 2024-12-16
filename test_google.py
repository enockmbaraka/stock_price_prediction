import requests


# Define the FastAPI endpoint URL
url = "http://127.0.0.1:8000/predict"

# Example time-series data (flat list)
test_time_series = [0.53865434, 0.55730858, 0.58422282, 0.58060334, 0.590348  ,
       0.60575407, 0.63517401, 0.63814392, 0.65039442, 0.63136901,
       0.52593977, 0.53577733, 0.54709985, 0.5591647 , 0.56306274,
       0.57642689, 0.57986086, 0.60853832, 0.59489562, 0.57271465,
       0.58013927, 0.55071933, 0.52983766, 0.53540602, 0.54867756,
       0.56296984, 0.56176344, 0.5024595 , 0.51461725, 0.49141539,
       0.51071935, 0.4984687 , 0.46329475, 0.45698377, 0.44519719,
       0.4728539 , 0.48241306, 0.50338749, 0.51109052, 0.52306262,
       0.553782  , 0.53596284, 0.59628767, 0.59025525, 0.60612538,
       0.59554534, 0.62496528, 0.61846879, 0.62403715, 0.62589327,
       0.62645009, 0.6687704 , 0.66013923, 0.66348031, 0.62273785,
       0.64102092, 0.66283073, 0.67907203, 0.67480282, 0.70515089]


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
