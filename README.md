# stock_price_prediction
Stock Price Prediction

# Table of Contents

1. [Stock Market Price Estimation](#stock-market-price-estimation)
2. [Data Collection](#data-collection)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data](#data)
6. [Model](#model)
7. [Training](#training)
8. [Deployment](#deployment)
9. [Containerization using Docker](#containerization)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)


## Stock Market Price Estimation

Stock market price estimation is a crucial aspect of financial analysis that impacts investors, companies, and analysts alike. The goal is to achieve accurate and reliable results to guide decision-making and minimize risks.

Inaccurate estimations can lead to:

- **Investor Losses:** Misleading predictions may cause poor investment decisions.
- **Market Panic:** Overly pessimistic forecasts can trigger unnecessary sell-offs.
- **Reputational Risk:** Financial institutions or analysts may lose credibility.

My aim is to analyze stock market price data for Google, Meta, Apple, and NVIDIA between October 30, 2021, and October 30, 2024, and provide an estimator while deploying my model both locally and to the cloud.

My project focuses on predicting the stock market's closing price for the 61st day based on the preceding 60 days of data. This provides an estimated future stock price, which can assist individuals in evaluating investment opportunities. Users can compare the predicted prices of companies such as Google, Meta, Apple, or NVIDIA to make more informed decisions about where to invest.

## [Data Collection](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/downloading_relevant_data.ipynb)

I used the [ ```yfinance```](https://github.com/ranaroussi/yfinance/tree/main?tab=readme-ov-file) library to download stock market price data for four companies—Google, Meta, Apple, and NVIDIA—from October 30, 2021, to October 30, 2024. The yfinance library is a Python tool for accessing historical market data, financial metrics, and company information from [Yahoo!Ⓡ finance](https://finance.yahoo.com/).

You can download the data in two steps as outlined below:


1) Specify the companies.

```tickers = ['GOOGL', 'META', 'AAPL', 'NVDA']```

2) Decide on the start and end dates, then download the data.

```stock_data = yf.download(tickers, start="2021-10-30", end="2024-10-30", group_by="ticker")```



 " ```yfinance``` is not affiliated, endorsed, or vetted by Yahoo, Inc. It's an open-source tool that uses Yahoo's publicly available APIs, and is intended for research and educational purposes. You should refer to Yahoo!'s terms of use ([here](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html), [here](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html), and [here](https://policies.yahoo.com/us/en/yahoo/terms/index.htm)) for details on your rights to use the actual data downloaded."



## Stock Price Dataset Description

This dataset contains daily stock price information, which is useful for analyzing stock performance, predicting trends, or conducting financial research.

| **Column Name** | **Description**                                                                                         |
|------------------|-------------------------------------------------------------------------------------------------------|
| **Date**         | The date of the record in the format `YYYY-MM-DD`.                                                   |
| **Open**         | The stock's opening price at the start of the trading day.                                           |
| **High**         | The highest price the stock reached during the trading day.                                          |
| **Low**          | The lowest price the stock fell to during the trading day.                                           |
| **Close**        | The stock's closing price at the end of the trading day.                                             |
| **Adj Close**    | The closing price adjusted for corporate actions like stock splits, dividends, and rights offerings. |
| **Volume**       | The total number of shares traded during the trading day. This reflects market activity and liquidity.|

In my project, I chose to analyze the __Close__ value from the stock price prediction dataset because it is widely regarded as one of the most reliable indicators for stock market analysis. The close value represents the final price at which a stock is traded at the end of the trading session, making it a critical metric for evaluating a stock's performance over time.

Unlike the open, low, or high values, which reflect specific moments or ranges during the trading day, the close value encapsulates the market's sentiment and activity for the entire trading session. It is often used by investors and analysts as a benchmark for decision-making, as it provides a clearer snapshot of how the stock performed on a given day.

## EDA

### Time Series Analysis Comments for Google, Meta, Apple, and NVIDIA

__1. Google__

__Trend:__ The trend is consistently increasing, reflecting a steady growth pattern in the time series over the given period. This could indicate positive long-term growth or increased activity.
__Seasonality:__ The seasonal component exhibits moderate fluctuations, showing periodic patterns that are relatively consistent. This suggests some underlying cyclical behavior in the data.
__Residual:__ The residuals are small and do not show significant spikes, indicating that the model captures the trend and seasonality well. The minimal noise may explain why I achieved the best score for Google.

__2. Meta__

__Trend:__ The trend is upward and fairly strong, indicating a significant increase in values over time. This suggests positive momentum in the dataset.
__Seasonality:__ Meta's seasonal component appears more volatile compared to Google, with frequent and irregular fluctuations. This suggests that the data has a more complex periodic structure.
__Residual:__ The residuals show more variability and larger deviations compared to Google, meaning that the model may not fully capture all aspects of the time series.

__3. Apple__

__Trend:__ The trend is upward and consistent but slightly less steep than NVIDIA and Meta. It indicates steady growth over time.
__Seasonality:__ Apple’s seasonal component shows relatively higher variability and irregular periodic patterns. This complexity in seasonality could make modeling more challenging.
__Residual:__ The residuals exhibit higher noise, suggesting that the model struggles to explain the variation in the data. This aligns with the  observation that Apple's classification results are the worst.

__4. NVIDIA__

__Trend:__ The trend for NVIDIA shows a strong and consistent upward movement, similar to Meta but with steeper growth. This indicates rapid changes or increases in the series.
__Seasonality:__ NVIDIA's seasonal component is less volatile than Apple or Meta, showing periodic fluctuations that are smoother and more predictable.
__Residual:__ The residuals are relatively well-contained but exhibit some spikes, indicating occasional deviations from the model’s predictions.

## Training

### Models
#### Recurrent Neural Networks (RNNs)

RNNs are designed to process sequential data by maintaining a memory of past inputs. This makes them a natural choice for time series, where patterns over time are critical. However, they struggle with long-term dependencies due to the vanishing gradient problem. While simpler and faster to train than some advanced models, their limited capacity to capture long-range relationships can be a drawback for complex time series.

__Advantages:__

- Simple architecture that captures short-term dependencies well.
- Computationally less expensive compared to more complex models.

__Disadvantages:__

- Struggles with long-term dependencies.
- Prone to vanishing gradient issues, leading to poorer performance on long sequences.

#### Neural Networks (NNs)

Standard neural networks (e.g., feedforward networks) are less commonly used in time series because they don't inherently account for sequential information. They treat each input as independent, which may lead to loss of critical temporal patterns unless engineered features are explicitly provided.

__Advantages:__

- Simpler to implement and train for non-sequential data or aggregated features.
- May perform well when time-dependent relationships are less critical.

__Disadvantages:__

- Does not natively capture sequential dependencies in the data.
- Requires manual feature engineering to represent time-based patterns effectively.
 
#### Long Short-Term Memory Networks (LSTMs)

LSTMs extend RNNs by incorporating memory cells and gates to selectively remember or forget information. This makes them well-suited for time series with long-term dependencies. They have been widely used in applications like stock price prediction, weather forecasting, and anomaly detection.

__Advantages:__

- Handles both short-term and long-term dependencies effectively.
- Robust to vanishing gradients, enabling better learning over extended sequences.

__Disadvantages:__

- Higher computational complexity compared to RNNs.
- Requires more tuning and longer training times

#### Transformers

Transformers revolutionized natural language processing and are increasingly applied to time series. Their self-attention mechanism allows them to capture both short-term and long-term dependencies efficiently. Transformers excel in handling irregular sampling and multivariate time series, making them powerful but computationally demanding.

__Advantages:__

- Can model long-term dependencies effectively with the self-attention mechanism.
- Handles multivariate time series and irregular time steps well.

__Disadvantages:__

- Computationally intensive, especially for large datasets or high-dimensional data.
- Requires large datasets for effective training, which can be a limitation for some time series problems.

#### Ordinary Differential Equations (ODEs)

ODE-based models are a different beast altogether. Instead of treating the data as discrete points, they model continuous changes in time, making them especially useful for time series where smooth dynamics are essential (e.g., physical systems, population growth, or epidemiology). Neural ODEs combine ODEs with neural networks, offering a flexible yet interpretable framework.

__Advantages:__

- Provides a continuous perspective on time, which is valuable for smooth or physics-inspired time series.
- More interpretable in scientific contexts, connecting directly to underlying processes.

__Disadvantages:__

- Computationally intensive to solve, especially for stiff ODEs.
- Requires domain knowledge for meaningful parameterization.

### [Hyperparameter Tuning](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/tree/main/parameter_tuning)

I performed hyperparameter tuning separately for [Google](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/parameter_tuning/hyperparameter_tuning_google_last.ipynb), [Meta](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/parameter_tuning/hyperparameter_tuning_meta_last.ipynb), [Apple](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/parameter_tuning/hyperparameter_tuning_apple_last.ipynb), and [NVIDIA](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/parameter_tuning/hyperparameter_tuning_nvidia_last.ipynb) using Optuna, an efficient hyperparameter optimization framework. Optuna was chosen for its ability to perform automated and flexible optimization through Bayesian search strategies. Unlike grid or random search, Optuna dynamically adjusts its search based on previous results, helping to find optimal hyperparameters more effectively and reducing computational overhead. 

During both hyperparameter tuning and training, I rescaled the time series data using Min-Max normalization. Min-Max normalization scales the data to a fixed range, typically [0, 1], which is particularly useful for time series since the values are often continuous and can vary across different magnitudes. 

### Model Training 

__Why Use Early Stopping?__
Early stopping is a regularization technique that helps prevent overfitting by halting the training process when the model's performance on the validation set stops improving. This ensures that the model generalizes well to unseen data rather than memorizing patterns from the training data.

__Why Choose a Patience Value of 15?__
A patience value of 15 strikes a balance between ensuring the model has enough time to converge and avoiding unnecessary training when improvements have plateaued. A smaller patience value (e.g., 5 or 10) might terminate training prematurely, especially in time series tasks, where validation loss often fluctuates before stabilizing. By allowing up to 15 epochs of no improvement, the model has the opportunity to fine-tune its weights and potentially discover a better solution.

__Training Configuration__
The total number of training epochs was set to 50. However, due to early stopping, training often concluded before reaching the maximum epoch limit, saving computational resources and reducing the risk of overfitting.

__How can the model be trained?__
First, you have to activate virtual environment:

```venv\Scripts\activate``` 

You can train the models running the codes provided below.

[```python main_train_google.py```](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/main_train_google.py)

[```python main_train_meta.py```](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/main_train_meta.py)

[```python main_train_apple.py```](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/main_train_apple.py)

[```python main_train_nvidia.py```](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/main_train_nvidia.py)


__Results Summary__

| **Dataset** | **Mean Absolute Error (MAE)** | **Mean Absolute Percentage Error (MAPE)** | **Symmetric Mean Absolute Percentage Error (SMAPE)** |
|-------------|--------------------------------|------------------------------------------|-----------------------------------------------------|
| Google      | 0.0205                        | 2.7047                                   | 2.5505                                              |
| Meta        | 0.0304                        | 4.4251                                   | 4.6505                                              |
| Apple       | 0.1824                        | 20.1099                                  | 22.9494                                             |
| Nvidia      | 0.0436                        | 5.0240                                   | 4.4939                                              |

![](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/google_result.png)

For Meta, the predicted prices show some correlation with the actual prices but tend to lag behind the true movements. This delay suggests that the model may not be adequately capturing rapid fluctuations in stock prices. Using an LSTM model, which performed slightly worse in this case, could potentially lead to better results with further optimization.


![](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/meta_result.png)

For Apple, the predicted prices (dashed line) consistently underestimate the actual stock prices (solid line). This indicates a potential bias in the model, where it struggles to accurately capture higher price fluctuations.


![](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/apple_result.png)

For Nvidia, the predicted prices closely follow the actual stock prices, which is encouraging. However, there are instances where the model overshoots or undershoots.

![](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/nvidia_result.png)

## Deployment with FastAPI

FastAPI is faster than Flask because it is built on ASGI and uses asynchronous programming, allowing it to handle high-concurrency workloads more efficiently. Unlike Flask, FastAPI automatically generates interactive API documentation (e.g., Swagger UI), which saves time and simplifies collaboration. It also uses Python's type hints for automatic request validation, reducing the risk of errors, whereas Flask requires additional libraries or manual validation. While Flask is lightweight and flexible, FastAPI's modern design makes it better suited for building APIs that need to scale or handle real-time requests.

I chose to use FastAPI instead of Flask, which is commonly used in the Zoomcamp, and created a [sub-repository](https://github.com/f-kuzey-edes-huyal/fastapi-project) to demonstrate how to work with FastAPI. The deployment process is straightforward and begins with the [main.py](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/main.py) file, which serves as the FastAPI application.

To deploy the model, follow these steps:


- Activate your virtual environment.

  ```venv\Scripts\activate```
  
- Run the following command in the terminal to start the FastAPI server:

```uvicorn main:app --reload```
- To test the deployed model: Open a new terminal and activate the virtual environment using

  ![](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/fastapi_stock_price.png)
  
  ```venv\Scripts\activate```

- Use the [test_all_companies.py](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/test_all_companies.py) script to make requests to the API. You can do this by running:

```python test_all_companies.py```


![](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/testing_fastapi_all.png)

## Containerization using Docker
When I tried to use the requirements.txt file that I prepared in my Windows environment in a Linux-based Docker environment, I encountered compatibility issues. To resolve this, I copied the packages listed in the <code> requirements.txt </code> file, which I had created using <code> pip freeze > requirements.txt </code>, and asked ChatGPT to generate a new requirements.txt file. I then used the updated requirements.txt file suggested by ChatGPT.

First, build the Docker image for the application.


<code> docker build -t stockmarket . </code>

Then, run the Docker container to deploy the model.

To test your model: Activate the virtual environment in another terminal (```venv\Scripts\activate```): and test the model (```python test_all_companies.py```)

<code> docker run -p 8000:8000 stockmarket </code>

![](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/docker_running_fastapi.png)

Finally, push the Docker image to Docker Hub. This will allow anyone to download the image and test the model easily.

<code> docker login </code>

<code> docker tag stockmarket fkuzeyedeshuyal/stockmarket </code>

<code> docker push fkuzeyedeshuyal/stockmarket </code>

You can download the image using the command given below.

<code> docker pull fkuzeyedeshuyal/stockmarket:latest </code>

## Deployment to Cloud

I want to begin this section by expressing my gratitude to Ajith Punnakula for his help and patience throughout the deployment process. Without his support, I might have been overwhelmed by numerous timeout errors.

I deployed my model on AWS using the following steps:

- Go to EC2 Instances and click on Launch Instances.
- Choose an AMI Image that is eligible for the Free Tier.
- Create a Key Pair.
- After setting the inbound rules correctly, launch the instance.
  
![myimage-alt-tag2](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/final_inbound_rules.png)

![myimage-alt-tag1](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/final_instance_ip.png)

- Click on the created instance, then click Connect, and finally, click the Connect button again. This will open a Linux terminal

 - Write the following commands
   
   ```sudo yum update -y```

   ```sudo yum install -y docker```

   ```sudo service docker start```

   ```sudo usermod -a -G docker ec2-user```

   ```docker ps```




Next, I pulled the Docker image I had previously pushed to Docker Hub. After pulling the image, I ran it using Docker.


```docker pull fkuzeyedeshuyal/stockmarket:latest``` 

```docker run -p 8000:8000 fkuzeyedeshuyal/stockmarket:latest```


Initially, I encountered an error because the inbound security rules for my EC2 instance were not set correctly. You can see the error in the screenshot at the top of the result.

This error was resolved after properly configuring the inbound rules for my EC2 instance.

![myimage-alt-tag3](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/final_aws_running1.png)

![myimage-alt-tag4](https://github.com/f-kuzey-edes-huyal/stock_price_prediction/blob/main/results/final_aws_test.png)




## Installation

```git clone https://github.com/f-kuzey-edes-huyal/stock_price_prediction.git```

__To-Do List:__

- Find a way to efficiently dockerize the TensorFlow model.
- Evaluate the Docker image.
- Get a smaller image and try AWS deployment before meeting with AJ!
