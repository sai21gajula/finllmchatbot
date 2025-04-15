# S&P 500 Prediction System

A deep learning system for predicting S&P 500 price movements and direction using LSTM/GRU networks with attention mechanisms.

## Overview

This project aims to predict both the direction (up/down) and price level of the S&P 500 index for the next trading day. It uses dual neural network models:

1. **Direction Model**: A binary classifier that predicts whether the S&P 500 will move up or down
2. **Price Model**: A regression model that predicts the actual closing price

The system uses extensive feature engineering with technical indicators and sophisticated neural network architectures including attention mechanisms.

## Directory Structure


## Features

- Comprehensive technical indicator calculation (47+ indicators)
- LSTM/GRU neural networks with attention mechanisms
- Data preprocessing with log transformation and robust scaling
- Visualization of predictions with technical analysis context
- Error handling and fallback mechanisms

## Usage

To generate a prediction for the next trading day:

python predict_sp500.py

This will:

Load the trained models and scalers
Generate a prediction for the next trading day
Create a visualization of the prediction
Save the results to the results directory

Models
The system uses two main models:

Direction Model: LSTM/GRU network trained to classify market movement direction
Price Model: LSTM/GRU network trained to predict the actual closing price

Both models utilize sequence data with a 60-day lookback window and include attention mechanisms to focus on the most relevant parts of the price history.
Technical Indicators
The system calculates a comprehensive set of technical indicators, including:

Moving Averages (SMA, EMA)
Momentum Indicators (RSI, Stochastic, Williams %R)
Trend Indicators (MACD, Moving Average Crossovers)
Volatility Indicators (Bollinger Bands, ATR)
Volume Indicators (OBV, ADI, EMV)
Price Transformations (Log Returns, Percent Changes)

Performance
The models were optimized using Optuna and exhibit the following performance metrics:
Direction Model:

Accuracy: ~55-60%
Precision: ~56%
Recall: ~57%
F1 Score: ~56%

Price Model:

MAE: ~0.5%
RMSE: ~0.7%
Directional Accuracy: ~58%

Note: These values are approximate and depend on market conditions.
Requirements

Python 3.8+
TensorFlow 2.x
NumPy
Pandas
scikit-learn
matplotlib
TA-Lib
joblib

License
This project is licensed under the MIT License - see the LICENSE file for details.