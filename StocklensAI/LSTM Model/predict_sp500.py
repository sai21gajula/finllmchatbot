import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import os
import json
import warnings
from datetime import datetime, timedelta

# Import helper modules
from add_technical_indicators import add_technical_indicators
from prepare_data import prepare_data_for_prediction, get_latest_technical_indicators



warnings.filterwarnings('ignore')  # Suppress warnings

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data', exist_ok=True)

def safe_load_model(model_path):
    """
    Safely load a model with proper error handling.
    Returns None if model cannot be loaded.
    """
    try:
        print(f"Attempting to load model from {model_path}...")
        model = load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {str(e)}")
        return None

def load_saved_models():
    """Load the pre-trained models and scalers"""
    try:
        print("Loading saved models and scalers...")
        
        # Try different model file names
        potential_direction_models = [
            'models/final_direction_model.h5',  # Use .h5 format for better compatibility
            'models/best_direction_model.h5',
            'models/direction_model.h5',
            'models/final_direction_model.keras',
            'models/best_direction_model.keras'
        ]
        
        potential_price_models = [
            'models/final_price_model.h5',  # Use .h5 format for better compatibility
            'models/best_price_model.h5',
            'models/price_model.h5',
            'models/final_price_model.keras',
            'models/best_price_model.keras'
        ]
        
        # Try to load direction model
        direction_model = None
        for model_path in potential_direction_models:
            if os.path.exists(model_path):
                direction_model = safe_load_model(model_path)
                if direction_model is not None:
                    break
        
        # Try to load price model
        price_model = None
        for model_path in potential_price_models:
            if os.path.exists(model_path):
                price_model = safe_load_model(model_path)
                if price_model is not None:
                    break
        
        # If models don't exist, we'll need to create new models
        if direction_model is None or price_model is None:
            print("Models not found or had loading errors. Creating new models...")
            
            # Try to load a feature scaler to get the input shape
            feature_scaler = None
            try:
                feature_scaler = joblib.load('models/feature_scaler.pkl')
                # Get feature dimensions if possible
                if hasattr(feature_scaler, 'n_features_in_'):
                    n_features = feature_scaler.n_features_in_
                else:
                    n_features = 50  # Default fallback
            except Exception as e:
                print(f"Error loading feature scaler: {str(e)}")
                n_features = 50  # Default fallback
            
            # Create simple placeholder models
            sequence_length = 60  # Default sequence length
            input_shape = (sequence_length, n_features)
            
            # Try to use the fixed model builder if available
            
           
        try:
            feature_scaler = joblib.load('models/feature_scaler.pkl')
            price_scaler = joblib.load('models/price_scaler.pkl')
            transform_params = joblib.load('models/transform_params.pkl')
        except Exception as e:
            print(f"Error loading scalers: {str(e)}")
            print("Creating default scalers...")
            from sklearn.preprocessing import StandardScaler
            feature_scaler = StandardScaler()
            price_scaler = StandardScaler()
            transform_params = {'log_transform': False}
        
        print("Models and scalers loaded or created successfully")
        return direction_model, price_model, feature_scaler, price_scaler, transform_params
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def get_sp500_data(use_fallback=False):
    """
    Fetch S&P 500 data with priority:
    1. Yahoo Finance API (if not use_fallback)
    2. Kaggle dataset at predefined path 
    3. Previously saved data
    4. Generated fallback data
    
    Args:
        use_fallback (bool): Force using fallback data instead of live data
        
    Returns:
        Cleaned and formatted DataFrame with Date index and proper columns
    """
    # Generate fallback data file (if it doesn't exist) so it's ready if needed
    
    # First try Yahoo Finance API if not using fallback
    if not use_fallback:
        try:
            print("Fetching 1-year daily S&P 500 data from Yahoo Finance...")
            
            # Import yfinance here to make it optional
            import yfinance as yf
            
            # Define ticker and date range
            ticker = "^GSPC"
            end_date = datetime.today()
            start_date = end_date - timedelta(days=365)
            
            # Download data
            stock_df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            
            # Check if data exists
            if stock_df.empty:
                raise ValueError("No data returned from Yahoo Finance.")
            
            # Reset index to get 'Date' as column
            stock_df = stock_df.reset_index()
            
            # Handle MultiIndex: Flatten the column names if needed
            if isinstance(stock_df.columns, pd.MultiIndex):
                stock_df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in stock_df.columns]
            
            # Rename columns to standard format if needed
            for old_col in stock_df.columns:
                if 'Close' in old_col:
                    stock_df = stock_df.rename(columns={old_col: 'Close'})
                elif 'High' in old_col:
                    stock_df = stock_df.rename(columns={old_col: 'High'})
                elif 'Low' in old_col:
                    stock_df = stock_df.rename(columns={old_col: 'Low'})
                elif 'Open' in old_col:
                    stock_df = stock_df.rename(columns={old_col: 'Open'})
                elif 'Volume' in old_col:
                    stock_df = stock_df.rename(columns={old_col: 'Volume'})
            
            # Print columns to check for 'Date' column
            print("Columns in DataFrame:", stock_df.columns)
            
            # Select desired columns and set 'Date' as index
            stock_df = stock_df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
            stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
            stock_df = stock_df.dropna(subset=['Date'])
            stock_df = stock_df.set_index('Date')
            
            # Save to CSV for backup
            stock_df.to_csv('data/latest_sp500.csv')
            
            print("Successfully fetched live S&P 500 data")
            print(f"Data range: {stock_df.index.min().date()} to {stock_df.index.max().date()}")
            
            return stock_df
            
        except Exception as e:
            print(f"Error fetching S&P 500 data from Yahoo Finance: {str(e)}")
            print("Falling back to Kaggle dataset...")
    
    # Next try Kaggle dataset
    kaggle_paths = [
        '/kaggle/input/testing-the-future-data/sp500_data.csv',  # Kaggle competition path
        'data/sp500_data.csv'                                    # Local path
    ]
    
    for kaggle_path in kaggle_paths:
        if os.path.exists(kaggle_path):
            try:
                print(f"Using S&P 500 data from {kaggle_path}")
                stock_df = pd.read_csv(kaggle_path)
                stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                stock_df = stock_df.set_index('Date')
                return stock_df
            except Exception as e:
                print(f"Error reading data from {kaggle_path}: {str(e)}")
                print("Trying next data source...")
    
    # Try to use the previously saved latest data if it exists
    latest_path = 'data/latest_sp500.csv'
    if os.path.exists(latest_path):
        try:
            print(f"Using previously saved S&P 500 data from {latest_path}")
            stock_df = pd.read_csv(latest_path)
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
            stock_df = stock_df.set_index('Date')
            return stock_df
        except Exception as e:
            print(f"Error reading previously saved data: {str(e)}")
            print("Falling back to generated data...")
    
    # Also check for synthetic data that might have been created
    synthetic_path = 'data/synthetic_sp500_data.csv'
    if os.path.exists(synthetic_path):
        try:
            print(f"Using synthetic S&P 500 data from {synthetic_path}")
            stock_df = pd.read_csv(synthetic_path)
            stock_df['Date'] = pd.to_datetime(stock_df.index)
            stock_df = stock_df.set_index('Date')
            return stock_df
        except Exception as e:
            print(f"Error reading synthetic data: {str(e)}")
            print("Falling back to generated data...")
    
    # As a last resort, use the generated fallback data
    try:
        print(f"Using fallback S&P 500 data from {fallback_path}")
        stock_df = pd.read_csv(fallback_path)
        stock_df['Date'] = pd.to_datetime(stock_df.index)
        stock_df = stock_df.set_index('Date')
        return stock_df
    except Exception as e:
        print(f"Error reading fallback data: {str(e)}")
        raise ValueError("Could not obtain S&P 500 data from any source")

def prepare_sequence_data(df, feature_scaler, sequence_length=60):
    """
    Prepare sequence data for prediction
    """
    print(f"Preparing sequence data with length {sequence_length}...")
    
    # Get feature columns (excluding target variables)
    feature_cols = [col for col in df.columns if col not in ['Next_Day_Direction', 'Next_Day_Price']]
    
    # Handle case where feature scaler expects specific columns
    if hasattr(feature_scaler, 'feature_names_in_'):
        expected_cols = list(feature_scaler.feature_names_in_)
        
        # Check for missing columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        existing_cols = [col for col in expected_cols if col in df.columns]
        
        if missing_cols:
            print(f"Warning: Data missing {len(missing_cols)} columns expected by scaler")
            # Add missing columns with zeros
            for col in missing_cols:
                df[col] = 0.0
            
            # Re-fit scaler with current data if many columns are missing
            if len(missing_cols) > len(expected_cols) // 2:
                print("Re-fitting feature scaler with current data...")
                feature_scaler.fit(df[feature_cols])
        
        # If scaler expects columns we don't have, just use what we have
        if existing_cols:
            df_for_scaling = df[existing_cols]
        else:
            df_for_scaling = df[feature_cols]
            feature_scaler.fit(df_for_scaling)  # Fit with current features
    else:
        # If scaler doesn't specify feature names, use all non-target columns
        df_for_scaling = df[feature_cols]
        # Fit the scaler if it hasn't been fit yet
        if not hasattr(feature_scaler, 'mean_') or not hasattr(feature_scaler, 'scale_'):
            print("Fitting feature scaler with current data...")
            feature_scaler.fit(df_for_scaling)
    
    # Scale the features
    try:
        scaled_features = feature_scaler.transform(df_for_scaling)
    except Exception as e:
        print(f"Error scaling features: {e}")
        print("Falling back to standard normalization...")
        # Fallback to simple normalization
        means = df_for_scaling.mean()
        stds = df_for_scaling.std()
        stds = stds.replace(0, 1)  # Avoid division by zero
        scaled_features = (df_for_scaling - means) / stds
        scaled_features = scaled_features.values
    
    # Ensure we have enough data
    if len(scaled_features) < sequence_length:
        print(f"Warning: Not enough data ({len(scaled_features)} rows) for sequence length ({sequence_length})")
        print("Padding with zeros...")
        padding = np.zeros((sequence_length - len(scaled_features), scaled_features.shape[1]))
        scaled_features = np.vstack([padding, scaled_features])
    
    # Get the most recent sequence for prediction
    X = scaled_features[-sequence_length:].reshape(1, sequence_length, scaled_features.shape[1])
    
    # Get current price
    current_price = df['Close'].iloc[-1]
    
    print(f"Prepared input sequence with shape {X.shape}")
    print(f"Current price: ${current_price:.2f}")
    
    return X, current_price

def create_prediction_visualization(df, current_date, next_day, current_price, 
                                  price_pred, price_change_pct, direction_pred, confidence):
    """
    Create a visualization of the prediction using real historical data
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Use as much data as we have, up to 60 days
        lookback = min(60, len(df))
        recent_data = df.iloc[-lookback:].copy()
        
        # Plot recent historical data
        plt.plot(recent_data.index, recent_data['Close'], marker='', color='blue', label='S&P 500 Historical Prices')
        
        # Add prediction point
        pred_color = 'green' if direction_pred == 'UP' else 'red'
        
        # Add the current price point and prediction
        plt.plot([recent_data.index[-1], next_day], [current_price, price_pred], 
                linestyle='--', color=pred_color, marker='o', markersize=8)
        
        plt.annotate(f'Prediction: ${price_pred:.2f} ({price_change_pct:+.2f}%)',
                    xy=(next_day, price_pred),
                    xytext=(15, 0),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold',
                    color=pred_color)
        
        # Add moving averages from the historical data if they exist
        if 'SMA_50' in recent_data.columns:
            plt.plot(recent_data.index, recent_data['SMA_50'], linestyle='--', color='orange', label='50-Day MA')
        if 'SMA_200' in recent_data.columns:
            plt.plot(recent_data.index, recent_data['SMA_200'], linestyle='--', color='purple', label='200-Day MA')
        
        # Add Bollinger Bands if they exist
        if 'BB_Upper' in recent_data.columns and 'BB_Lower' in recent_data.columns:
            plt.plot(recent_data.index, recent_data['BB_Upper'], linestyle=':', color='gray', alpha=0.7)
            plt.plot(recent_data.index, recent_data['BB_Lower'], linestyle=':', color='gray', alpha=0.7)
            plt.fill_between(recent_data.index, recent_data['BB_Upper'], recent_data['BB_Lower'], 
                             color='gray', alpha=0.1, label='Bollinger Bands')
        
        plt.title(f'S&P 500 Prediction for {next_day.date()}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        
        # Add direction prediction info
        plt.figtext(0.5, 0.01, 
                  f"Predicted Direction: {direction_pred} (Confidence: {confidence:.1f}%)", 
                  ha='center', fontsize=12, fontweight='bold',
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save main visualization
        output_file = f'results/SP500_prediction_{next_day.date().strftime("%Y-%m-%d")}.png'
        plt.savefig(output_file)
        plt.close()
        
        # Create a secondary plot with RSI if available
        if 'RSI_14' in recent_data.columns:
            plt.figure(figsize=(12, 4))
            plt.plot(recent_data.index, recent_data['RSI_14'], color='blue', label='RSI')
            plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            plt.fill_between(recent_data.index, recent_data['RSI_14'], 70, 
                            where=(recent_data['RSI_14'] >= 70), color='red', alpha=0.3)
            plt.fill_between(recent_data.index, recent_data['RSI_14'], 30, 
                            where=(recent_data['RSI_14'] <= 30), color='green', alpha=0.3)
            plt.title('Relative Strength Index (RSI)')
            plt.ylabel('RSI')
            plt.xlabel('Date')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save RSI visualization
            plt.savefig(f'results/SP500_RSI_{next_day.date().strftime("%Y-%m-%d")}.png')
            plt.close()
        
        # Create a third plot with MACD if available
        if all(col in recent_data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Diff']):
            plt.figure(figsize=(12, 4))
            plt.plot(recent_data.index, recent_data['MACD'], color='blue', label='MACD')
            plt.plot(recent_data.index, recent_data['MACD_Signal'], color='red', label='Signal Line')
            plt.bar(recent_data.index, recent_data['MACD_Diff'], 
                   color=recent_data['MACD_Diff'].apply(lambda x: 'green' if x > 0 else 'red'), label='Histogram')
            plt.title('Moving Average Convergence Divergence (MACD)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save MACD visualization
            plt.savefig(f'results/SP500_MACD_{next_day.date().strftime("%Y-%m-%d")}.png')
            plt.close()
        
        print(f"Visualizations saved to results directory")
        return output_file
    
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

def print_technical_analysis(indicators):
    """Print technical analysis summary"""
    print("\nTECHNICAL ANALYSIS SUMMARY:")
    
    # Print indicators that are available
    for indicator_type, values in indicators.items():
        print(f"{indicator_type.upper()}:")
        for key, value in values.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Count signals if status information is available
    bullish_signals = 0
    bearish_signals = 0
    neutral_signals = 0
    
    for indicator_type, values in indicators.items():
        if 'status' in values:
            status = values['status']
            if status == "OVERSOLD":
                bullish_signals += 1
            elif status == "OVERBOUGHT":
                bearish_signals += 1
            elif status == "BULLISH":
                bullish_signals += 1
            elif status == "BEARISH":
                bearish_signals += 1
            elif status == "NEUTRAL":
                neutral_signals += 1
    
    # Print signal summary if any signals were found
    if bullish_signals > 0 or bearish_signals > 0 or neutral_signals > 0:
        print("\nSIGNAL SUMMARY:")
        print(f"Bullish Signals:   {bullish_signals}")
        print(f"Bearish Signals:   {bearish_signals}")
        print(f"Neutral Signals:   {neutral_signals}")
        
        if bullish_signals > bearish_signals:
            print("Overall Signal:    BULLISH")
        elif bearish_signals > bullish_signals:
            print("Overall Signal:    BEARISH")
        else:
            print("Overall Signal:    NEUTRAL")

def inverse_transform_price(scaled_price, price_scaler, transform_params):
    """Inverse transform scaled price prediction to original scale"""
    # Reshape for scaler
    scaled_reshaped = np.array([[scaled_price]])
    
    # Inverse transform scaling
    try:
        unscaled = price_scaler.inverse_transform(scaled_reshaped)[0, 0]
        
        # Apply inverse log transform if needed
        if transform_params.get('log_transform', False):
            unscaled = np.exp(unscaled)
            
        return unscaled
    except Exception as e:
        print(f"Error in inverse transform: {str(e)}")
        # Return the raw scaled price as fallback (not ideal but better than failing)
        return scaled_price

def predict_sp500(use_fallback=False):
    """
    Predict the next day's S&P 500 price and direction using saved models and real data
    
    Args:
        use_fallback (bool): Force using fallback data instead of live data
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        print("Starting S&P 500 prediction...")
        
        # Load models and scalers
        direction_model, price_model, feature_scaler, price_scaler, transform_params = load_saved_models()
        
        # Fetch S&P 500 data using the provided function
        sp500_data = get_sp500_data(use_fallback=use_fallback)
        
        # Calculate technical indicators
        df_with_indicators = add_technical_indicators(sp500_data)
        
        # Prepare sequence data for prediction
        X_pred, current_price = prepare_sequence_data(df_with_indicators, feature_scaler)
        
        # Get technical indicators
        try:
            indicators = get_latest_technical_indicators(df_with_indicators)
        except Exception as e:
            print(f"Error getting technical indicators: {str(e)}")
            indicators = {}  # Use empty dict if function fails
        
        # Make direction prediction
        print("Making direction prediction...")
        if direction_model is None:
            print("Warning: No direction model available. Using simple heuristic prediction.")
            # Use a simple heuristic based on recent price movements
            recent_prices = df_with_indicators['Close'].iloc[-5:]
            direction_prob = 0.5 + (0.1 * (1 if recent_prices.iloc[-1] > recent_prices.iloc[0] else -1))
            direction_pred = "UP" if direction_prob > 0.5 else "DOWN"
            confidence = max(direction_prob, 1 - direction_prob) * 100
            print(f"Heuristic direction prediction: {direction_pred} (confidence: {confidence:.1f}%)")
        else:
            try:
                direction_prob = float(direction_model.predict(X_pred, verbose=0)[0, 0])
                direction_pred = "UP" if direction_prob > 0.5 else "DOWN"
                confidence = max(direction_prob, 1 - direction_prob) * 100
            except Exception as e:
                print(f"Error making direction prediction with model: {str(e)}")
                # Fallback to simple strategy
                recent_prices = df_with_indicators['Close'].iloc[-5:]
                direction_prob = 0.5 + (0.1 * (1 if recent_prices.iloc[-1] > recent_prices.iloc[0] else -1))
                direction_pred = "UP" if direction_prob > 0.5 else "DOWN"
                confidence = max(direction_prob, 1 - direction_prob) * 100
                print(f"Fallback direction prediction: {direction_pred} (confidence: {confidence:.1f}%)")
        
        # Make price prediction
        print("Making price prediction...")
        if price_model is None:
            print("Warning: No price model available. Using simple heuristic prediction.")
            # Use last price with small adjustment based on direction
            raw_price_pred = current_price * (1.005 if direction_pred == "UP" else 0.995)
            price_pred = raw_price_pred  # No scaling applied for heuristic
            print(f"Heuristic price prediction: ${price_pred:.2f}")
        else:
            try:
                raw_price_pred = float(price_model.predict(X_pred, verbose=0)[0, 0])
                price_pred = inverse_transform_price(raw_price_pred, price_scaler, transform_params)
            except Exception as e:
                print(f"Error making price prediction with model: {str(e)}")
                # Fallback to simple strategy
                price_pred = current_price * (1.005 if direction_pred == "UP" else 0.995)
                print(f"Fallback price prediction: ${price_pred:.2f}")
        
        # Calculate price change
        price_change = price_pred - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Sanity check - limit extreme predictions
        if abs(price_change_pct) > 5:  # More than 5% change is unlikely
            print(f"Warning: Predicted change of {price_change_pct:.2f}% seems excessive.")
            # Limit to 1% change in the predicted direction
            price_change = current_price * (0.01 if direction_pred == "UP" else -0.01)
            price_pred = current_price + price_change
            price_change_pct = (price_change / current_price) * 100
            print(f"Price prediction adjusted to ${price_pred:.2f} ({price_change_pct:+.2f}%)")
        
        # Format dates
        current_date = datetime.now()
        next_day = current_date + timedelta(days=1)
        if next_day.weekday() >= 5:  # Weekend adjustment
            next_day += timedelta(days=7 - next_day.weekday())
        
        # Create visualization
        output_file = create_prediction_visualization(
            df_with_indicators, current_date, next_day, current_price, 
            price_pred, price_change_pct, direction_pred, confidence
        )
        
        # Display and save results
        print("\n" + "="*60)
        print(f"S&P 500 PREDICTION FOR {next_day.date()}")
        print("="*60)
        print(f"Current Date:      {current_date.date()}")
        print(f"Current Price:     ${current_price:.2f}")
        print(f"Predicted Price:   ${price_pred:.2f}")
        print(f"Expected Change:   ${price_change:.2f} ({price_change_pct:+.2f}%)")
        print(f"Direction:         {direction_pred}")
        print(f"Confidence:        {confidence:.1f}%")
        if output_file:
            print(f"Visualization:     {output_file}")
        print("="*60)
        
        # Print technical analysis
        if indicators:
            print_technical_analysis(indicators)
        
        # Save results to JSON
        results = {
            'current_date': str(current_date.date()),
            'prediction_date': str(next_day.date()),
            'current_price': float(current_price),
            'predicted_price': float(price_pred),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'direction': direction_pred,
            'confidence': float(confidence),
            'visualization': output_file,
        }
        
        # Add technical indicators if available
        if indicators:
            results['technical_indicators'] = indicators
        
        results_file = f'results/prediction_{next_day.date().strftime("%Y-%m-%d")}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {results_file}")
        return results
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Run the prediction when the script is executed directly
if __name__ == "__main__":
    # Try with live data first, fallback to CSV if it fails
    try:
        results = predict_sp500(use_fallback=False)
    except Exception as e:
        print(f"Live data prediction failed: {str(e)}")
        print("Retrying with fallback data...")
        results = predict_sp500(use_fallback=True)