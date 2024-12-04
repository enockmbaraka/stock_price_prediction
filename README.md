# stock_price_prediction
Stock Price Prediction



## Stock Market Price Estimation

Stock market price estimation is a crucial aspect of financial analysis that impacts investors, companies, and analysts alike. The goal is to achieve accurate and reliable results to guide decision-making and minimize risks.

Inaccurate estimations can lead to:

- **Investor Losses:** Misleading predictions may cause poor investment decisions.
- **Market Panic:** Overly pessimistic forecasts can trigger unnecessary sell-offs.
- **Reputational Risk:** Financial institutions or analysts may lose credibility.



## Data collection

I used the [ ```yfinance```](https://github.com/ranaroussi/yfinance/tree/main?tab=readme-ov-file) library to download stock market price data for four companies—Google, Meta, Apple, and NVIDIA—from October 30, 2021, to October 30, 2024.

[Yahoo!Ⓡ finance](https://finance.yahoo.com/).

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


## Installation

```git clone https://github.com/f-kuzey-edes-huyal/stock_price_prediction.git```

