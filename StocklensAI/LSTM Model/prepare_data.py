import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
import os

def prepare_data_for_prediction(df, feature_scaler, sequence_length=60):
    """Prepare data for prediction using the pre-trained models"""
    print("Preparing data for prediction...")
    
    # Define feature columns based on scaler
    if hasattr(feature_scaler, 'feature_names_in_'):
        feature_cols = feature_scaler.feature_names_in_.tolist()
        print(f"Using {len(feature_cols)} features from scaler")
    else:
        feature_cols = df.columns.tolist()
        print(f"Using all {len(feature_cols)} columns as features")
    
    # Ensure all feature columns exist in the dataframe
    for col in feature_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in data. Adding with zeros.")
            df[col] = 0
    
    # Scale features
    scaled_features = feature_scaler.transform(df[feature_cols]).astype(np.float32)
    
    # Take the last sequence_length rows of data
    if len(scaled_features) < sequence_length:
        print(f"Warning: Not enough data ({len(scaled_features)} rows). Padding with zeros.")
        padding = np.zeros((sequence_length - len(scaled_features), len(feature_cols)))
        sequence_data = np.vstack([padding, scaled_features])
    else:
        sequence_data = scaled_features[-sequence_length:]
    
    # Create sequence for model input (batch_size=1, sequence_length, n_features)
    X = np.zeros((1, sequence_length, len(feature_cols)), dtype=np.float32)
    X[0] = sequence_data
    
    print(f"Input sequence shape: {X.shape}")
    return X, feature_cols

def prepare_sequence_data(df, feature_scaler, sequence_length=60):
    """
    Prepare sequence data for prediction - simplified version of prepare_data_for_prediction
    that returns current price as well
    """
    print(f"Preparing sequence data with length {sequence_length}...")
    
    # Get current price
    current_price = df['Close'].iloc[-1]
    
    # Use the more comprehensive function
    X, _ = prepare_data_for_prediction(df, feature_scaler, sequence_length)
    
    print(f"Current price: ${current_price:.2f}")
    
    return X, current_price

def inverse_transform_price(scaled_price, price_scaler, transform_params):
    """Inverse transform price prediction to original scale"""
    # Reshape for scaler
    scaled_reshaped = np.array(scaled_price).reshape(-1, 1)
    
    # Inverse transform scaling
    unscaled = price_scaler.inverse_transform(scaled_reshaped)
    
    # Inverse transform log if used
    if transform_params.get('log_transform', True):
        return np.exp(unscaled).flatten()[0]
    else:
        return unscaled.flatten()[0]

def prepare_data_sequences(train_data, valid_data, test_data, sequence_length=60, prediction_horizon=1, 
                          use_log_transform=True, use_robust_scaler=True):
    """
    Prepare sequence data for LSTM/GRU models with both price and direction prediction.
    Enhanced with log transformation and robust scaling options for price data.
    """
    print(f"Preparing data with sequence length {sequence_length}...")
    train_data = train_data.copy()
    valid_data = valid_data.copy()
    test_data = test_data.copy()
    
    # Add target variables if they don't exist
    for df in [train_data, valid_data, test_data]:
        if 'Next_Day_Direction' not in df.columns:
            df['Next_Day_Direction'] = (df['Close'].shift(-prediction_horizon) > df['Close']).astype(int)
        if 'Next_Day_Price' not in df.columns:
            df['Next_Day_Price'] = df['Close'].shift(-prediction_horizon)
    
    # Drop NaN values
    train_data = train_data.dropna()
    valid_data = valid_data.dropna()
    test_data = test_data.dropna()
    
    # Define columns for targets and features
    direction_col = 'Next_Day_Direction'
    price_col = 'Next_Day_Price'
    feature_cols = [col for col in train_data.columns if col not in [direction_col, price_col]]
    
    # Scale features - standard scaler works well for most financial features
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_data[feature_cols])
    
    scaled_train_features = feature_scaler.transform(train_data[feature_cols]).astype(np.float32)
    scaled_valid_features = feature_scaler.transform(valid_data[feature_cols]).astype(np.float32)
    scaled_test_features = feature_scaler.transform(test_data[feature_cols]).astype(np.float32)
    
    # Process price data with enhanced techniques
    if use_log_transform:
        print("Using log transformation for price data")
        # Apply log transform to price data (ensuring all values are positive)
        train_prices = np.log(train_data[[price_col]].values)
        valid_prices = np.log(valid_data[[price_col]].values)
        test_prices = np.log(test_data[[price_col]].values)
    else:
        train_prices = train_data[[price_col]].values
        valid_prices = valid_data[[price_col]].values
        test_prices = test_data[[price_col]].values
    
    # Use appropriate scaler for price data
    if use_robust_scaler:
        print("Using RobustScaler for price data")
        price_scaler = RobustScaler()
    else:
        print("Using MinMaxScaler for price data")
        price_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit price scaler only on training data
    price_scaler.fit(train_prices)
    
    # Transform price data
    scaled_train_prices = price_scaler.transform(train_prices).flatten().astype(np.float32)
    scaled_valid_prices = price_scaler.transform(valid_prices).flatten().astype(np.float32)
    scaled_test_prices = price_scaler.transform(test_prices).flatten().astype(np.float32)
    
    # Save the scalers for future prediction
    os.makedirs('models', exist_ok=True)
    joblib.dump(feature_scaler, 'models/feature_scaler.pkl')
    joblib.dump(price_scaler, 'models/price_scaler.pkl')
    
    # Store transformation parameters for later inverse transform
    transform_params = {
        'log_transform': use_log_transform,
        'robust_scaler': use_robust_scaler
    }
    joblib.dump(transform_params, 'models/transform_params.pkl')
    
    print("Scalers and transform parameters saved to models directory")
    
    # Create sequences for LSTM training
    def create_sequences(features, direction_values, price_values, seq_length):
        total_sequences = len(features) - seq_length - prediction_horizon + 1
        X = np.zeros((total_sequences, seq_length, features.shape[1]), dtype=np.float32)
        y_dir = np.zeros(total_sequences, dtype=np.int32)
        y_price = np.zeros(total_sequences, dtype=np.float32)
        
        for i in range(total_sequences):
            X[i] = features[i:i+seq_length]
            idx = i + seq_length + prediction_horizon - 1
            if idx < len(direction_values):
                y_dir[i] = direction_values[idx]
                y_price[i] = price_values[idx]
        
        return X, y_dir, y_price
    
    train_dir = train_data[direction_col].values
    valid_dir = valid_data[direction_col].values
    test_dir = test_data[direction_col].values
    
    X_train, y_train_dir, y_train_price = create_sequences(scaled_train_features, train_dir, scaled_train_prices, sequence_length)
    X_val, y_val_dir, y_val_price = create_sequences(scaled_valid_features, valid_dir, scaled_valid_prices, sequence_length)
    X_test, y_test_dir, y_test_price = create_sequences(scaled_test_features, test_dir, scaled_test_prices, sequence_length)
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Compute class weights for direction prediction
    total_samples = len(y_train_dir)
    n_positive = np.sum(y_train_dir)
    n_negative = total_samples - n_positive
    class_weight = {
        0: total_samples / (2.0 * n_negative),
        1: total_samples / (2.0 * n_positive)
    }
    print(f"Class distribution - Up: {n_positive}/{total_samples} ({n_positive/total_samples:.2f}), "
          f"Down: {n_negative}/{total_samples} ({n_negative/total_samples:.2f})")
    print(f"Class weights: {class_weight}")
    
    return (X_train, y_train_dir, y_train_price,
            X_val, y_val_dir, y_val_price,
            X_test, y_test_dir, y_test_price,
            feature_scaler, price_scaler, feature_cols, class_weight, transform_params)

def get_latest_technical_indicators(df):
    """
    Extract the latest technical indicators for analysis
    """
    # Get the latest row
    latest = df.iloc[-1]
    
    # Create indicators dictionary
    indicators = {
        'rsi': {
            'value': float(latest['RSI_14']),
            'period': 14
        },
        'macd': {
            'value': float(latest['MACD']),
            'signal': float(latest['MACD_Signal']),
            'histogram': float(latest['MACD_Diff'])
        },
        'sma': {
            'short_period': 50,
            'short_value': float(latest['SMA_50']),
            'long_period': 200,
            'long_value': float(latest['SMA_200'])
        },
        'bollinger': {
            'upper': float(latest['BB_Upper']),
            'middle': float(latest['SMA_20']),
            'lower': float(latest['BB_Lower']),
            'width': float(latest['BB_Width']),
            'period': 20
        },
        'stochastic': {
            'k': float(latest['Stoch_k']),
            'd': float(latest['Stoch_d'])
        },
        'atr': {
            'value': float(latest['ATR_14']),
            'period': 14
        },
        'momentum': {
            '1d': float(latest['Price_Change']),
            '5d': float(latest['Close'] / df['Close'].iloc[-6] - 1) if len(df) > 5 else 0,
            '10d': float(latest['Close'] / df['Close'].iloc[-11] - 1) if len(df) > 10 else 0,
            '20d': float(latest['Close'] / df['Close'].iloc[-21] - 1) if len(df) > 20 else 0
        },
        'volume': {
            'change': float(latest['Volume_Change']),
            'ratio': float(latest['Volume_Ratio_5'])
        }
    }
    
    # Add indicator statuses
    # RSI
    if indicators['rsi']['value'] > 70:
        indicators['rsi']['status'] = "OVERBOUGHT"
    elif indicators['rsi']['value'] < 30:
        indicators['rsi']['status'] = "OVERSOLD"
    else:
        indicators['rsi']['status'] = "NEUTRAL"
    
    # MACD
    indicators['macd']['status'] = "BULLISH" if indicators['macd']['value'] > indicators['macd']['signal'] else "BEARISH"
    
    # SMA Crossover
    indicators['sma']['status'] = "BULLISH" if indicators['sma']['short_value'] > indicators['sma']['long_value'] else "BEARISH"
    
    # Bollinger Bands
    price = latest['Close']
    if price > indicators['bollinger']['upper']:
        indicators['bollinger']['status'] = "OVERBOUGHT"
    elif price < indicators['bollinger']['lower']:
        indicators['bollinger']['status'] = "OVERSOLD"
    else:
        indicators['bollinger']['status'] = "NEUTRAL"
        
    # Stochastic
    if indicators['stochastic']['k'] > 80 and indicators['stochastic']['d'] > 80:
        indicators['stochastic']['status'] = "OVERBOUGHT"
    elif indicators['stochastic']['k'] < 20 and indicators['stochastic']['d'] < 20:
        indicators['stochastic']['status'] = "OVERSOLD"
    else:
        indicators['stochastic']['status'] = "NEUTRAL"
    
    # Volume
    if indicators['volume']['ratio'] > 1.5:
        indicators['volume']['status'] = "HIGH"
    elif indicators['volume']['ratio'] < 0.5:
        indicators['volume']['status'] = "LOW"
    else:
        indicators['volume']['status'] = "NORMAL"
    
    return indicators

# Test if run directly
if __name__ == "__main__":
    print("Testing data preparation utilities...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=300, freq='B')
    data = pd.DataFrame({
        'Open': np.random.normal(450, 5, 300),
        'High': np.random.normal(455, 5, 300),
        'Low': np.random.normal(445, 5, 300),
        'Close': np.random.normal(450, 5, 300),
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)
    
    # Add some basic features
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['RSI_14'] = np.random.uniform(30, 70, 300)
    
    # Create train/valid/test splits
    train_data = data[:200]
    valid_data = data[200:250]
    test_data = data[250:]
    
    # Test prepare_data_sequences
    sequence_data = prepare_data_sequences(
        train_data, valid_data, test_data, 
        sequence_length=60, 
        prediction_horizon=1
    )
    
    # Load and test prediction preparation
    feature_scaler = joblib.load('models/feature_scaler.pkl')
    X_pred, feature_cols = prepare_data_for_prediction(test_data, feature_scaler, sequence_length=60)
    print("Prediction input shape:", X_pred.shape)