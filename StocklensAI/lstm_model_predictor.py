import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
import traceback
from datetime import datetime, timedelta
import importlib.util

def lstm_model_predictor_section():
    st.markdown('<div class="page-title">S&P 500 LSTM Predictor</div>', unsafe_allow_html=True)
    setup_path()

    # Welcome Banner
    st.markdown("""
        <div class="welcome-banner">
            <h3>Advanced S&P 500 Prediction with LSTM Neural Networks</h3>
            <p>Our deep learning model combines technical indicators and historical patterns to predict market movements. Get tomorrow's price and direction predictions with confidence scores.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    # Function to import predict_sp500 dynamically
    def import_prediction_module():
        # Try multiple possible locations
        possible_paths = [
            "LSTM Model/predict_sp500.py",
            "predict_sp500.py",
            "models/predict_sp500.py"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location("predict_sp500", path)
                predict_sp500_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(predict_sp500_module)
                return predict_sp500_module
        
        st.error("Could not find predict_sp500.py in any expected location")
        return None
    
    with col1:
        # Settings panel
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Prediction Settings</div>', unsafe_allow_html=True)
        
        # Data source selection
       # Data source selection
        data_source = st.radio(
        "Data Source",
        ["Live Yahoo Finance Data", "Use Local CSV Data"],
        key="lstm_data_source"  # Add this unique key
        )
        
        use_fallback = False  # Default to live data
        
        if data_source == "Use Local CSV Data":
            use_fallback = True
            data_files = []
            
            # Check both potential data locations
            data_paths = ["data", "LSTM Model/data"]
            for path in data_paths:
                if os.path.exists(path):
                    for file in os.listdir(path):
                        if file.endswith(".csv"):
                            data_files.append(f"{path}/{file}")
            
            if data_files:
                st.selectbox("Available Data Files", data_files, key="data_file_info")
                st.info("The selected file will be used if live data fails")
            else:
                st.warning("No CSV files found in data directories.")
        
        # Run prediction button
        predict_button = st.button("🚀 Run S&P 500 Prediction", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model info panel
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Model Information</div>', unsafe_allow_html=True)
        
        # Find model files in multiple potential locations
        # Find model files in multiple potential locations
        model_locations = ["models", "LSTM Model/models", os.path.join(os.getcwd(), "LSTM Model/models")]
        model_files = {
         "Direction Model": ["direction_model.keras", "final_direction_model.keras", "best_direction_model.keras"],
        "Price Model": ["price_model.keras", "final_price_model.keras", "best_price_model.keras"],
        "Feature Scaler": ["feature_scaler.pkl"],
        "Price Scaler": ["price_scaler.pkl"]
        }
        
        # Check for model files in the specified locations

        model_statuses = {}
        for model_name, file_list in model_files.items():
            model_found = False
            for location in model_locations:
                for file in file_list:
                    full_path = os.path.join(location, file) if "/" not in location else location + "/" + file
                    if os.path.exists(full_path):
                        model_found = True
                        st.session_state[f"found_{model_name.lower().replace(' ', '_')}"] = full_path
                        break
                if model_found:
                    break
            model_statuses[model_name] = model_found



        # Display model availability
        for model_name, status in model_statuses.items():
            status_color = "#10b981" if status else "#ef4444"  # Green if exists, red if not
            status_text = "Available" if status else "Not Found"
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span>{model_name}:</span>
                    <span style="color: {status_color}; font-weight: 600;">{status_text}</span>
                </div>
            """, unsafe_allow_html=True)
        
        if all(model_statuses.values()):
            with st.expander("Model Architecture"):
                st.markdown("""
                    <p style="font-size: 0.9rem;">
                        <strong>LSTM Architecture:</strong> Dual model system with attention mechanism<br>
                        <strong>Features:</strong> 50+ technical indicators<br>
                        <strong>Direction Model:</strong> Binary classifier (Up/Down)<br>
                        <strong>Price Model:</strong> Regression model for price prediction
                    </p>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="card-title" style="margin-top: 1rem;">Market Information</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div style="font-size: 0.9rem;">
                <p><strong>Current Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
                <p><strong>Market Status:</strong> {"Open" if datetime.now().weekday() < 5 and 9 <= datetime.now().hour <= 16 else "Closed"}</p>
                <p><strong>Prediction Target:</strong> Next trading day</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Results area
        if predict_button:
            try:
                # Show loading indicator
                with st.spinner("🔮 Running LSTM prediction models... This may take a moment."):
                    # Import prediction module
                    predict_sp500_module = import_prediction_module()
                    
                    if predict_sp500_module is None:
                        st.error("Failed to import prediction module")
                    else:
                        # Call the prediction function with our parameters
                        result = predict_sp500_module.predict_sp500(use_fallback=use_fallback)
                
                # Display prediction results
                if result:
                    # Success message
                    st.markdown("""
                        <div class="success-box">
                            <h4 style="margin-top: 0; color: #10b981;">✅ Prediction Completed</h4>
                            <p>The LSTM model has successfully generated predictions for the next trading day.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display prediction card
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">S&P 500 Prediction Results</div>', unsafe_allow_html=True)
                    
                    # Extract key metrics from result
                    current_price = result.get('current_price', 0)
                    predicted_price = result.get('predicted_price', 0)
                    price_change = result.get('price_change', 0)
                    price_change_pct = result.get('price_change_pct', 0)
                    direction = result.get('direction', 'NEUTRAL')
                    confidence = result.get('confidence', 0)
                    prediction_date = result.get('prediction_date', datetime.now().strftime('%Y-%m-%d'))
                    
                    # Display the metrics in a nice grid
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    
                    # Current Price
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Current Price</div>
                            <div class="metric-value">${current_price:.2f}</div>
                            <div class="metric-change">{datetime.now().strftime('%Y-%m-%d')}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Predicted Price
                    prediction_color = "#10b981" if price_change >= 0 else "#ef4444"
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Predicted Price</div>
                            <div class="metric-value" style="color: {prediction_color}">${predicted_price:.2f}</div>
                            <div class="metric-change" style="color: {prediction_color}">
                                {'+' if price_change >= 0 else ''}{price_change:.2f} ({price_change_pct:+.2f}%)
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Direction
                    direction_color = "#10b981" if direction == "UP" else "#ef4444" if direction == "DOWN" else "#6366f1"
                    direction_icon = "↗️" if direction == "UP" else "↘️" if direction == "DOWN" else "↔️"
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Predicted Direction</div>
                            <div class="metric-value" style="color: {direction_color}">{direction_icon} {direction}</div>
                            <div class="metric-change">Confidence: {confidence:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Date
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Prediction For</div>
                            <div class="metric-value" style="font-size: 1.4rem;">{prediction_date}</div>
                            <div class="metric-change">Next trading day</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display prediction visualization if available
                    if 'visualization' in result:
                        vis_path = result['visualization']
                        if os.path.exists(vis_path):
                            st.image(vis_path, caption="S&P 500 Prediction Visualization", use_column_width=True)
                        else:
                            # Try alternative paths
                            alt_paths = [
                                f"LSTM Model/{vis_path}",
                                f"results/SP500_prediction_{prediction_date.replace('-', '_')}.png"
                            ]
                            
                            vis_found = False
                            for alt_path in alt_paths:
                                if os.path.exists(alt_path):
                                    st.image(alt_path, caption="S&P 500 Prediction Visualization", use_column_width=True)
                                    vis_found = True
                                    break
                                    
                            if not vis_found:
                                st.warning(f"Visualization file not found. Check the results directory.")
                    
                    # Show technical indicators if available
                    if 'technical_indicators' in result:
                        st.markdown('<div class="card-title">Technical Analysis</div>', unsafe_allow_html=True)
                        
                        indicators = result['technical_indicators']
                        
                        # Create tabs for different indicator categories
                        tab1, tab2, tab3, tab4 = st.tabs(["Trend", "Momentum", "Volatility", "Volume"])
                        
                        with tab1:
                            # Trend indicators
                            if 'sma' in indicators:
                                sma = indicators['sma']
                                sma_status = sma.get('status', 'NEUTRAL')
                                status_color = "#10b981" if sma_status == "BULLISH" else "#ef4444" if sma_status == "BEARISH" else "#6366f1"
                                
                                st.markdown(f"""
                                    <div style="margin-bottom: 1rem;">
                                        <strong>SMA Crossover:</strong> <span style="color: {status_color}">{sma_status}</span><br>
                                        SMA-50: ${sma.get('short_value', 0):.2f}<br>
                                        SMA-200: ${sma.get('long_value', 0):.2f}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            if 'macd' in indicators:
                                macd = indicators['macd']
                                macd_status = macd.get('status', 'NEUTRAL')
                                status_color = "#10b981" if macd_status == "BULLISH" else "#ef4444" if macd_status == "BEARISH" else "#6366f1"
                                
                                st.markdown(f"""
                                    <div style="margin-bottom: 1rem;">
                                        <strong>MACD:</strong> <span style="color: {status_color}">{macd_status}</span><br>
                                        MACD Line: {macd.get('value', 0):.4f}<br>
                                        Signal Line: {macd.get('signal', 0):.4f}<br>
                                        Histogram: {macd.get('histogram', 0):.4f}
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        with tab2:
                            # Momentum indicators
                            if 'rsi' in indicators:
                                rsi = indicators['rsi']
                                rsi_value = rsi.get('value', 0)
                                rsi_status = rsi.get('status', 'NEUTRAL')
                                status_color = "#10b981" if rsi_status == "OVERSOLD" else "#ef4444" if rsi_status == "OVERBOUGHT" else "#6366f1"
                                
                                st.markdown(f"""
                                    <div style="margin-bottom: 1rem;">
                                        <strong>RSI-14:</strong> {rsi_value:.2f} <span style="color: {status_color}">({rsi_status})</span>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # RSI Chart (simplified visualization)
                                fig, ax = plt.subplots(figsize=(8, 2))
                                # Create a gauge-like visualization for RSI
                                ax.barh(0, 100, height=0.4, color='#e2e8f0')
                                ax.barh(0, rsi_value, height=0.4, color='#3b82f6')
                                ax.axvline(x=30, color='#10b981', linestyle='--', alpha=0.7)
                                ax.axvline(x=70, color='#ef4444', linestyle='--', alpha=0.7)
                                ax.set_xlim(0, 100)
                                ax.set_ylim(-0.5, 0.5)
                                ax.text(0, 0, "0", ha='center', va='center', fontsize=8)
                                ax.text(30, 0, "30", ha='center', va='center', fontsize=8)
                                ax.text(70, 0, "70", ha='center', va='center', fontsize=8)
                                ax.text(100, 0, "100", ha='center', va='center', fontsize=8)
                                ax.text(rsi_value, 0, f"{rsi_value:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
                                ax.set_yticks([])
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            if 'stochastic' in indicators:
                                stoch = indicators['stochastic']
                                stoch_status = stoch.get('status', 'NEUTRAL')
                                status_color = "#10b981" if stoch_status == "OVERSOLD" else "#ef4444" if stoch_status == "OVERBOUGHT" else "#6366f1"
                                
                                st.markdown(f"""
                                    <div style="margin-bottom: 1rem;">
                                        <strong>Stochastic:</strong> <span style="color: {status_color}">{stoch_status}</span><br>
                                        %K: {stoch.get('k', 0):.2f}<br>
                                        %D: {stoch.get('d', 0):.2f}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            if 'momentum' in indicators:
                                momentum = indicators['momentum']
                                
                                st.markdown(f"""
                                    <div style="margin-bottom: 1rem;">
                                        <strong>Price Momentum:</strong><br>
                                        1-Day: {momentum.get('1d', 0)*100:+.2f}%<br>
                                        5-Day: {momentum.get('5d', 0)*100:+.2f}%<br>
                                        10-Day: {momentum.get('10d', 0)*100:+.2f}%
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        with tab3:
                            # Volatility indicators
                            if 'bollinger' in indicators:
                                bollinger = indicators['bollinger']
                                bb_status = bollinger.get('status', 'NEUTRAL')
                                status_color = "#10b981" if bb_status == "OVERSOLD" else "#ef4444" if bb_status == "OVERBOUGHT" else "#6366f1"
                                
                                st.markdown(f"""
                                    <div style="margin-bottom: 1rem;">
                                        <strong>Bollinger Bands:</strong> <span style="color: {status_color}">{bb_status}</span><br>
                                        Upper Band: ${bollinger.get('upper', 0):.2f}<br>
                                        Middle Band: ${bollinger.get('middle', 0):.2f}<br>
                                        Lower Band: ${bollinger.get('lower', 0):.2f}<br>
                                        Width: {bollinger.get('width', 0):.4f}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            if 'atr' in indicators:
                                atr = indicators['atr']
                                
                                st.markdown(f"""
                                    <div style="margin-bottom: 1rem;">
                                        <strong>ATR-{atr.get('period', 14)}:</strong> ${atr.get('value', 0):.2f}
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        with tab4:
                            # Volume indicators
                            if 'volume' in indicators:
                                volume = indicators['volume']
                                volume_status = volume.get('status', 'NORMAL')
                                status_color = "#10b981" if volume_status == "HIGH" else "#ef4444" if volume_status == "LOW" else "#6366f1"
                                
                                st.markdown(f"""
                                    <div style="margin-bottom: 1rem;">
                                        <strong>Volume Analysis:</strong> <span style="color: {status_color}">{volume_status}</span><br>
                                        Volume Change: {volume.get('change', 0)*100:+.2f}%<br>
                                        Volume Ratio (5-day): {volume.get('ratio', 0):.2f}
                                    </div>
                                """, unsafe_allow_html=True)
                    
                    # Overall signal from technical analysis
                    if 'technical_indicators' in result:
                        # Count bullish vs bearish signals
                        bullish_signals = 0
                        bearish_signals = 0
                        neutral_signals = 0
                        
                        for indicator_type, values in result['technical_indicators'].items():
                            if 'status' in values:
                                status = values['status']
                                if status == "OVERSOLD" or status == "BULLISH":
                                    bullish_signals += 1
                                elif status == "OVERBOUGHT" or status == "BEARISH":
                                    bearish_signals += 1
                                elif status == "NEUTRAL":
                                    neutral_signals += 1
                        
                        # Determine overall signal
                        overall_signal = "NEUTRAL"
                        signal_color = "#6366f1"  # Default purple for neutral
                        
                        if bullish_signals > bearish_signals:
                            overall_signal = "BULLISH"
                            signal_color = "#10b981"  # Green
                        elif bearish_signals > bullish_signals:
                            overall_signal = "BEARISH"
                            signal_color = "#ef4444"  # Red
                        
                        # Display the overall signal
                        st.markdown(f"""
                            <div style="margin-top: 1rem; padding: 1rem; background-color: rgba(99, 102, 241, 0.1); border-radius: 8px; text-align: center;">
                                <h3 style="margin: 0; color: {signal_color};">Overall Signal: {overall_signal}</h3>
                                <div style="margin-top: 0.5rem;">
                                    <span style="color: #10b981; margin-right: 1rem;">Bullish: {bullish_signals}</span>
                                    <span style="color: #ef4444; margin-right: 1rem;">Bearish: {bearish_signals}</span>
                                    <span style="color: #6366f1;">Neutral: {neutral_signals}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional visualizations if available
                    prediction_date = result.get('prediction_date', '')
                    if prediction_date:
                        # Format the date correctly for filenames
                        try:
                            date_obj = datetime.strptime(prediction_date, "%Y-%m-%d")
                            date_formatted = date_obj.strftime("%Y-%m-%d")
                            
                            # Check multiple possible locations
                            possible_paths = [
                                f"results/SP500_RSI_{date_formatted}.png",
                                f"results/SP500_MACD_{date_formatted}.png",
                                f"LSTM Model/results/SP500_RSI_{date_formatted}.png",
                                f"LSTM Model/results/SP500_MACD_{date_formatted}.png"
                            ]
                            
                            # Find which files exist
                            available_plots = [path for path in possible_paths if os.path.exists(path)]
                            
                            # Display if any are found
                            if available_plots:
                                st.markdown("<h4>Additional Technical Analysis Charts</h4>", unsafe_allow_html=True)
                                
                                # Organize by type
                                rsi_plots = [p for p in available_plots if "RSI" in p]
                                macd_plots = [p for p in available_plots if "MACD" in p]
                                
                                if rsi_plots or macd_plots:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if rsi_plots:
                                            st.image(rsi_plots[0], caption="RSI Indicator", use_column_width=True)
                                    
                                    with col2:
                                        if macd_plots:
                                            st.image(macd_plots[0], caption="MACD Indicator", use_column_width=True)
                        except:
                            pass  # Silently handle date parsing errors
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Disclaimer
                    st.markdown("""
                        <div style="margin-top: 1rem; padding: 0.75rem; background-color: #f8fafc; border-radius: 8px; font-size: 0.8rem; color: #64748b; text-align: center;">
                            <p style="margin: 0;">Disclaimer: These predictions are generated by machine learning models and should not be considered financial advice. Past performance is not indicative of future results.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown("""
                        <div class="error-box">
                            <h4 style="margin-top: 0; color: #ef4444;">⚠️ Prediction Failed</h4>
                            <p>The model was unable to generate predictions. Please check the logs for more information.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                    <div class="error-box">
                        <h4 style="margin-top: 0; color: #ef4444;">⚠️ Error</h4>
                        <p>An error occurred during prediction: {str(e)}</p>
                    </div>
                """, unsafe_allow_html=True)
                st.error(traceback.format_exc())
        else:
            # Default view when button not clicked

            # Default view when button not clicked

            # Default view when button not clicked
            st.write("### S&P 500 LSTM Prediction")
            st.write("This advanced deep learning model uses Long Short-Term Memory (LSTM) neural networks to predict S&P 500 movements. The model is trained on historical market data and over 50 technical indicators.")
                        
            st.write("**The model predicts two key outputs:**")
            st.markdown("""
            - **Market Direction:** Whether the S&P 500 will move UP or DOWN the next trading day
            - **Price Target:** The expected closing price for the next trading day
            """)
                        
            st.write("Click the \"Run S&P 500 Prediction\" button to generate a fresh prediction using the latest market data.")

            st.write("### How The Model Works")
            st.write("Our prediction system uses a multi-step process:")
                        
            st.markdown("""
            1. **Data Collection:** Obtaining S&P 500 historical price data
            2. **Feature Engineering:** Calculating technical indicators (RSI, MACD, Bollinger Bands, etc.)
            3. **Sequence Creation:** Preparing the last 60 days of data as a sequence
            4. **Model Prediction:** Running the data through our dual LSTM models
            5. **Result Processing:** Converting model outputs to actual price predictions
            6. **Visualization:** Generating charts to explain the prediction
            """)
                        
            st.write("The models have been trained on historical market data spanning multiple years and market conditions.")

            st.write("### Best Practices")
            st.markdown("""
            - Use this model as one of several inputs in your investment decisions
            - Focus on the direction confidence score as a key metric
            - Pay attention to contradicting technical indicators
            - Remember that market predictions are inherently uncertain
            - For best results, run predictions before market open
            """)

def setup_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lstm_model_dir = os.path.join(base_dir, "LSTM Model")
    
    if os.path.exists(lstm_model_dir) and lstm_model_dir not in sys.path:
        sys.path.append(lstm_model_dir)
    
    try:
        try:
            import add_technical_indicators
        except ImportError:
            module_path = os.path.join(lstm_model_dir, "add_technical_indicators.py")
            if not os.path.exists(module_path):
                module_path = os.path.join(base_dir, "add_technical_indicators.py")
            
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location("add_technical_indicators", module_path)
                add_technical_indicators_module = importlib.util.module_from_spec(spec)
                sys.modules["add_technical_indicators"] = add_technical_indicators_module
                spec.loader.exec_module(add_technical_indicators_module)
        
        for module_name, filename in [
            ("prepare_data", "prepare_data.py"),
            ("fallback_data", "fallback_data.py")
        ]:
            try:
                __import__(module_name)
            except ImportError:
                module_path = os.path.join(lstm_model_dir, filename)
                if not os.path.exists(module_path):
                    module_path = os.path.join(base_dir, filename)
                
                if os.path.exists(module_path):
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
    
    except Exception as e:
        st.error(f"Error setting up modules: {str(e)}")

# This allows the file to be imported
if __name__ == "__main__":
    # Setup path for imports
    setup_path()
    
    # If run directly, create a minimal Streamlit app for testing
    st.set_page_config(
        layout="wide", 
        page_title="LSTM Model Predictor", 
        page_icon="📈"
    )
    
    lstm_model_predictor_section()