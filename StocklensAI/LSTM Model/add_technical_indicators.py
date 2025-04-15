import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, EaseOfMovementIndicator

def add_technical_indicators(data):
    """Add various technical indicators to the dataframe"""
    print("Adding technical indicators...")
    df = data.copy()

    # Price Transformations
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Change'] = df['Close'].pct_change()

    # Time-based features
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter

    # Trend Indicators
    df['SMA_5'] = SMAIndicator(close=df['Close'], window=5).sma_indicator()
    df['SMA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
    df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()

    df['EMA_5'] = EMAIndicator(close=df['Close'], window=5).ema_indicator()
    df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator()
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['EMA_200'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()

    # Deviation from moving averages (normalized)
    df['Deviation_SMA_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Deviation_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Deviation_SMA_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

    # Moving average crossover signals
    df['SMA_5_10_Crossover'] = (df['SMA_5'] > df['SMA_10']).astype(int)
    df['SMA_10_20_Crossover'] = (df['SMA_10'] > df['SMA_20']).astype(int)
    df['SMA_20_50_Crossover'] = (df['SMA_20'] > df['SMA_50']).astype(int)
    df['SMA_50_200_Crossover'] = (df['SMA_50'] > df['SMA_200']).astype(int)

    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    df['MACD_Crossover'] = ((df['MACD'] > df['MACD_Signal']).astype(int) -
                           (df['MACD'] < df['MACD_Signal']).astype(int))

    # Momentum Indicators
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['RSI_7'] = RSIIndicator(close=df['Close'], window=7).rsi()

    # Stochastic oscillator
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()
    df['Stoch_Crossover'] = ((df['Stoch_k'] > df['Stoch_d']).astype(int) -
                            (df['Stoch_k'] < df['Stoch_d']).astype(int))

    # Williams %R
    df['Williams_%R_14'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()

    # Volatility Indicators
    bb = BollingerBands(close=df['Close'])
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    df['ATR_14'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['ATR_Percent'] = df['ATR_14'] / df['Close'] * 100

    # Volume Indicators
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['ADI'] = AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).acc_dist_index()
    df['EMV'] = EaseOfMovementIndicator(high=df['High'], low=df['Low'], volume=df['Volume']).ease_of_movement()

    # Volume-based features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_SMA_5'] = SMAIndicator(close=df['Volume'], window=5).sma_indicator()
    df['Volume_Ratio_5'] = df['Volume'] / df['Volume_SMA_5']

    # Ichimoku Cloud
    try:
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'])
        df['Ichimoku_a'] = ichimoku.ichimoku_a()
        df['Ichimoku_b'] = ichimoku.ichimoku_b()
        df['Ichimoku_Trend'] = ((df['Close'] > df['Ichimoku_a']) & (df['Close'] > df['Ichimoku_b'])).astype(int)
    except Exception as e:
        print(f"Warning: Could not calculate Ichimoku Cloud: {e}")
        df['Ichimoku_a'] = df['Close']
        df['Ichimoku_b'] = df['Close']
        df['Ichimoku_Trend'] = 0

    # Target variable: Next day direction (1 if price goes up, 0 if price goes down)
    df['Next_Day_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Next day price
    df['Next_Day_Price'] = df['Close'].shift(-1)

    # Handle NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')

    print(f"Added {len(df.columns) - len(data.columns)} technical indicators")
    return df