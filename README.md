# stock_price_prediction
Stock Price Prediction

# Table of Contents

1. [Stock Market Price Estimation](#stock-market-price-estimation)
2. [Data Collection](#data-collection)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data](#data)
6. [Model](#model)
7. [Deployment](#deployment)
8. [Containerization using Docker](#containerization)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)


## Stock Market Price Estimation

Stock market price estimation is a crucial aspect of financial analysis that impacts investors, companies, and analysts alike. The goal is to achieve accurate and reliable results to guide decision-making and minimize risks.

Inaccurate estimations can lead to:

- **Investor Losses:** Misleading predictions may cause poor investment decisions.
- **Market Panic:** Overly pessimistic forecasts can trigger unnecessary sell-offs.
- **Reputational Risk:** Financial institutions or analysts may lose credibility.

My aim is to analyze stock market price data for Google, Meta, Apple, and NVIDIA between October 30, 2021, and October 30, 2024, and provide an estimator while deploying my model both locally and to the cloud.

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

### Hyperparameter Tuning

### Training 

```venv\Scripts\activate``` 

## Containerization using Docker
When I tried to use the requirements.txt file that I prepared in my Windows environment in a Linux-based Docker environment, I encountered compatibility issues. To resolve this, I copied the packages listed in the <code> requirements.txt </code> file, which I had created using <code> pip freeze > requirements.txt </code>, and asked ChatGPT to generate a new requirements.txt file. I then used the updated requirements.txt file suggested by ChatGPT.

<code> docker build -t stockmarket . </code>

<code> docker run -p 8000:8000 stockmarket </code>

<code> docker login </code>

<code> docker tag stockmarket   fkuzeyedeshuyal/stockmarket </code>

<code> docker push fkuzeyedeshuyal/stockmarket </code>


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
