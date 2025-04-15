import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from crew import run_analysis
import json
import pandas as pd
import numpy as np
import traceback
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from lstm_model_predictor import lstm_model_predictor_section
# Set page configuration 
st.set_page_config(
    layout="wide", 
    page_title="FinWisely: Your AI Financial Guide", 
    page_icon="💼",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling with better shadows, lighting and gradients
def add_custom_css():
    st.markdown("""
    <style>
        /* Global styling */
        .main {
            background-color: #f8fafc;
            color: #0f172a;
        }
        
        /* Sidebar styling with gradient */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding-top: 2rem;
        }
        
        [data-testid="stSidebar"] .sidebar-content {
            color: white;
        }
        
        /* Typography */
        h1 {
            color: #1e3a8a;
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.025em;
            margin-bottom: 1.5rem !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h2 {
            color: #1e40af;
            font-size: 2rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }
        
        h3 {
            color: #1e40af;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        .app-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .page-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #1e3a8a;
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
            background: linear-gradient(to right, #f8fafc, white, #f8fafc);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Cards with improved shadows and borders */
        .card {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 6px;
            height: 100%;
            background: linear-gradient(to bottom, #3b82f6, #1e3a8a);
            border-top-left-radius: 12px;
            border-bottom-left-radius: 12px;
        }
        
        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .card-content {
            color: #334155;
            line-height: 1.7;
        }
        
        /* Welcome banner with gradient and enhanced lighting */
        .welcome-banner {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 80%, #60a5fa 100%);
            border-radius: 12px;
            padding: 2rem;
            color: white;
            margin-bottom: 2.5rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            position: relative;
            overflow: hidden;
        }
        
        .welcome-banner::after {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 90%;
            height: 90%;
            background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 70%);
            transform: rotate(30deg);
            pointer-events: none;
        }
        
        .welcome-banner h3 {
            color: white !important;
            font-weight: 700 !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            margin-top: 0 !important;
        }
        
        .welcome-banner p {
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            font-size: 1.1rem;
            line-height: 1.6;
        }
        
        /* Analysis sections with better styling */
        .analysis-section {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }
        
        .analysis-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid #3b82f6;
        }
        
        /* Reports with gradient borders */
        .report-section {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            position: relative;
            overflow: hidden;
        }
        
        .report-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background: linear-gradient(to right, #3b82f6, #60a5fa);
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
        }
        
        .report-header {
            font-size: 1.75rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 1.5rem;
            text-align: center;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Enhanced buttons */
        .stButton>button {
            background: linear-gradient(to right, #1e3a8a, #3b82f6);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background: linear-gradient(to right, #1e40af, #1d4ed8);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
        }
        
        .stButton>button:active {
            transform: translateY(0);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        /* Input fields with better styling */
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            padding: 0.75rem 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
        }
        
        /* Metrics display with 3D effect */
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, white 0%, #f8fafc 100%);
            border-radius: 10px;
            padding: 1.25rem;
            flex: 1;
            min-width: 160px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        
        .metric-title {
            font-size: 0.875rem;
            color: #64748b;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.25rem;
            text-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .metric-change {
            font-size: 0.875rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .positive-change {
            color: #10b981;
        }
        
        .negative-change {
            color: #ef4444;
        }
        
        /* Status boxes with gradient borders */
        .success-box {
            background-color: rgba(16, 185, 129, 0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-left: 6px solid #10b981;
            position: relative;
            overflow: hidden;
        }
        
        .success-box::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 6px;
            background: linear-gradient(to bottom, #10b981, #059669);
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        
        .error-box {
            background-color: rgba(239, 68, 68, 0.1);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border-left: 6px solid #ef4444;
            position: relative;
            overflow: hidden;
        }
        
        .error-box::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 6px;
            background: linear-gradient(to bottom, #ef4444, #dc2626);
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        
        /* Chart container with depth effect */
        .chart-container {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            border: 1px solid #e2e8f0;
            overflow: hidden;
        }
        
        /* Footer with gradient */
        .footer {
            background: linear-gradient(to right, #f8fafc, white, #f8fafc);
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
            font-size: 0.875rem;
            box-shadow: 0 -4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        .footer p {
            margin-bottom: 0.5rem;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8fafc;
            border-radius: 8px;
            font-weight: 600;
            color: #1e3a8a;
            padding: 0.75rem 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: #f1f5f9;
        }
        
        .streamlit-expanderContent {
            border: 1px solid #e2e8f0;
            border-top: none;
            border-radius: 0 0 8px 8px;
            padding: 1.25rem;
            background-color: white;
        }
        
        /* Sidebar radio button styling */
        .stRadio > div {
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .stRadio > div > div > label {
            color: white !important;
            font-weight: 1300;
        }
        
        /* Section dividers with gradient */
        .divider {
            height: 4px;
            background: linear-gradient(to right, #3b82f6, #60a5fa, #93c5fd);
            border-radius: 2px;
            margin: 2rem 0;
        }
        
        /* Loading animation */
        .stSpinner > div > div {
            border-top-color: #3b82f6 !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #94a3b8;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #64748b;
        }

        /* Fix for dataframe display */
        .dataframe-container {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            font-size: 0.9rem;
        }
        
        .dataframe th {
            background-color: #1e3a8a;
            color: white;
            font-weight: 600;
            text-align: left;
            padding: 0.75rem 1rem;
            border: none;
        }
        
        .dataframe td {
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .dataframe tr:nth-child(even) {
            background-color: #f8fafc;
        }
        
        .dataframe tr:hover {
            background-color: #f1f5f9;
        }
    </style>
    """, unsafe_allow_html=True)

# Main Function
def main():
    add_custom_css()
    
    # Sidebar Configuration
    st.sidebar.markdown("""
        <div class="app-title">
            <h1 style="color: white; font-size: 2rem; margin-bottom: 0.5rem;">FinWisely</h1>
            <p style="opacity: 0.8;">AI-Powered Financial Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    
    # Navigation with improved styling
    st.sidebar.markdown("<h2 style='color: white; margin-top: 2rem;'>Navigation</h2>", unsafe_allow_html=True)
    options = st.sidebar.radio("", ["Stock Analysis", "LSTM Model Predictor", "StockLensAI"], key="main_navigation")

    # Main Content Area
    if options == "Stock Analysis":
        stock_analysis_section()
    elif options == "LSTM Model Predictor":
        lstm_model_predictor_section()
    elif options == "StockLensAI":
        stocklensai_section()
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p><strong>FinWisely</strong> © 2025 | AI-Powered Financial Analysis Platform</p>
            <p>Disclaimer: This platform provides analysis for educational purposes only. Not financial advice.</p>
        </div>
    """, unsafe_allow_html=True)

def stock_analysis_section():
    st.markdown('<div class="page-title">AI-Powered Stock Analysis</div>', unsafe_allow_html=True)
    
    # Welcome Banner
    st.markdown("""
        <div class="welcome-banner">
            <h3>Welcome to FinWisely Stock Analyzer</h3>
            <p>Harness the power of AI and multi-agent systems to analyze stocks from technical, fundamental, and sentiment perspectives. Get comprehensive reports generated by specialized AI agents working together.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create a layout with columns
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown('<h3>Analysis Settings</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Configure Analysis</div>', unsafe_allow_html=True)
        stock_symbol = st.text_input("Stock Symbol", value="^GSPC")
        time_period = st.selectbox("Time Period", ['3mo', '6mo', '1y', '2y', '5y'])
        
        indicators = st.multiselect("Technical Indicators", 
                                   ['Moving Averages', 'Volume', 'RSI', 'MACD'],
                                   default=['Moving Averages', 'Volume'])
        
        analyze_button = st.button("🚀 Analyze Stock", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a quick stats card if data available
        if 'stock_data' in st.session_state and not st.session_state.stock_data.empty:
            data = st.session_state.stock_data
            latest = data.iloc[-1]
            prev_day = data.iloc[-2]
            percent_change = ((latest['Close'] - prev_day['Close']) / prev_day['Close']) * 100
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">{stock_symbol} Quick Stats</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            
            # Current Price
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Current Price</div>
                    <div class="metric-value">${latest['Close'].iloc[0]:.2f}</div>
                    <div class="metric-change {'positive-change' if percent_change.iloc[0] >= 0 else 'negative-change'}">
                    {'+' if percent_change.iloc[0] >= 0 else ''}{percent_change.iloc[0]:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Volume
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Volume</div>
                 <div class="metric-value">{int(latest['Volume'].iloc[0]/1000)}K</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3>Analysis Results</h3>', unsafe_allow_html=True)
        
        if analyze_button:
            try:
                with st.spinner("Fetching market data..."):
                    stock_data = get_stock_data(stock_symbol, time_period)
                    st.session_state.stock_data = stock_data
                    
                if stock_data is not None and not stock_data.empty:
                    # Display chart with styling
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(plot_stock_chart(stock_data, indicators), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Proceed with agent analysis
                    with st.spinner("🧠 AI Agents are analyzing the stock... This might take a few minutes."):
                        analysis = perform_crew_analysis(stock_symbol)
                    
                    if analysis:
                        st.markdown("""
                            <div class="success-box">
                                <h4 style="margin-top: 0; color: #10b981;">✅ Analysis Complete</h4>
                                <p>Our AI agents have successfully analyzed the stock across technical, fundamental, and sentiment dimensions.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Display analysis results
                        display_analysis_report(analysis)
                else:
                    st.markdown(f"""
                        <div class="error-box">
                            <h4 style="margin-top: 0; color: #ef4444;">⚠️ Data Error</h4>
                            <p>Could not retrieve data for symbol {stock_symbol}. Please check if the symbol is correct.</p>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                    <div class="error-box">
                        <h4 style="margin-top: 0; color: #ef4444;">⚠️ Analysis Error</h4>
                        <p>An error occurred during analysis: {str(e)}</p>
                    </div>
                """, unsafe_allow_html=True)
                st.error(traceback.format_exc())
        else:
            # Default state message
            st.markdown("""
                <div class="card">
                    <div class="card-title">Start Your Analysis</div>
                    <div class="card-content">
                        <p>Enter a stock symbol and click the "Analyze Stock" button to begin.</p>
                        <p>Our AI agents will analyze the stock from multiple perspectives:</p>
                        <ul>
                            <li><strong>Technical Analysis:</strong> Price patterns, indicators, and trends</li>
                            <li><strong>Fundamental Analysis:</strong> Company financials, valuation metrics</li>
                            <li><strong>Sentiment Analysis:</strong> Market sentiment and news impact</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def display_analysis_report(analysis):
    """Enhanced function to display the analysis report with better formatting"""
    if not analysis:
        st.warning("No analysis data available.")
        return
    
    # Main report container
    st.markdown('<div class="report-header">Stock Analysis Report</div>', unsafe_allow_html=True)
    
    # Get the report content
    report_content = None
    
    if 'report' in analysis:
        # Extract report content based on its type
        if isinstance(analysis['report'], str):
            # If it's already a string, use it directly
            report_content = analysis['report']
        elif isinstance(analysis['report'], dict) and 'report' in analysis['report']:
            # If it's a dict with a 'report' key, extract that value
            report_content = analysis['report']['report']
        else:
            # Try to convert to string
            report_content = str(analysis['report'])
            
            # If it looks like JSON, try to parse it
            if report_content.startswith('{') and report_content.endswith('}'):
                try:
                    report_json = json.loads(report_content.replace("'", '"'))
                    if 'report' in report_json:
                        report_content = report_json['report']
                except json.JSONDecodeError:
                    # If JSON parsing fails, just use the string as is
                    pass
    
    # Clean up any formatting issues
    if report_content:
        # Remove any JSON artifacts
        if report_content.startswith('"') and report_content.endswith('"'):
            report_content = report_content[1:-1]
        
        # Unescape common escape sequences
        report_content = report_content.replace('\\n', '\n')
        report_content = report_content.replace('\\t', '\t')
        report_content = report_content.replace('\\"', '"')
        report_content = report_content.replace('\\\\', '\\')
        
        # Replace problematic characters
        report_content = report_content.replace("∗", "*")
        report_content = report_content.replace("−", "-")
        
        # Remove excessive spaces
        while "  " in report_content:
            report_content = report_content.replace("  ", " ")
        
        # Fix newline issues
        report_content = report_content.replace("\n\n\n", "\n\n")
        
        # Display the clean report as markdown in a container
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.markdown(report_content)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Show error message if no report content could be extracted
        st.error("Could not extract report content from the analysis data.")
        
        # Show raw analysis data for debugging
        with st.expander("Debug: Raw Analysis Data"):
            st.write(analysis)


            
def stocklensai_section():
    st.markdown('<div class="page-title">StockLensAI: Your Stock Research Assistant</div>', unsafe_allow_html=True)
    
    # Welcome Banner for StockLensAI
    st.markdown("""
        <div class="welcome-banner">
            <h3>Interactive Stock Research Assistant</h3>
            <p>Ask questions about your analyzed stocks and get AI-powered answers based on technical data, fundamental analysis, and market sentiment.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load previous analysis results if available
    results_path = "results"
    available_analyses = []
    
    if os.path.exists(results_path):
        for item in os.listdir(results_path):
            if os.path.isdir(os.path.join(results_path, item)):
                symbol = item
                if symbol not in available_analyses:
                    available_analyses.append(symbol)
    
    # Display available analyses
    if available_analyses:
        st.markdown("""
            <div class="card">
                <div class="card-title">Available Stock Analyses</div>
                <div class="card-content">
                    These stocks have already been analyzed and are ready for questions.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create columns for chat interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            selected_stock = st.selectbox("Select a stock:", available_analyses)
            
            # Show stock info if available
            latest_files = []
            if os.path.exists(os.path.join(results_path, selected_stock)):
                for file in os.listdir(os.path.join(results_path, selected_stock)):
                    if file.endswith('.json'):
                        latest_files.append(file)
            
            if latest_files:
                st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-card">
                            <div class="metric-title">Analysis Files</div>
                            <div class="metric-value">{len(latest_files)}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Last Analyzed</div>
                            <div class="metric-value" style="font-size: 1.2rem;">{latest_files[-1].split('_')[1].split('.')[0]}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Ask a Question</div>', unsafe_allow_html=True)
            
            # Query examples
            st.markdown("""
                <div style="margin-bottom: 1rem; font-size: 0.9rem;">
                    <strong>Example questions:</strong>
                    <ul style="margin-top: 0.5rem;">
                        <li>What are the support levels for this stock?</li>
                        <li>What is the P/E ratio?</li>
                        <li>What's the sentiment analysis result?</li>
                        <li>What are the key risks?</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            question = st.text_input("Your question:", placeholder="E.g., What are the support levels for this stock?")
            
            ask_button = st.button("🔍 Get Answer", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show answer area
        if question and ask_button:
            try:
                st.markdown('<div class="report-section">', unsafe_allow_html=True)
                st.markdown(f'<h3>Answer for {selected_stock}</h3>', unsafe_allow_html=True)
                
                # Simulate searching through analysis files
                with st.spinner("Searching through analysis data..."):
                    time.sleep(1)  # Simulate processing
                    
                    all_files = []
                    if os.path.exists(os.path.join(results_path, selected_stock)):
                        for file in os.listdir(os.path.join(results_path, selected_stock)):
                            if file.endswith('.json'):
                                all_files.append(os.path.join(results_path, selected_stock, file))
                    
                    if all_files:
                        # Get the most recent files for each type
                        latest_technical = None
                        latest_fundamental = None
                        latest_research = None
                        latest_report = None
                        
                        for file in sorted(all_files):
                            if 'technical' in file and (latest_technical is None or file > latest_technical):
                                latest_technical = file
                            elif 'fundamental' in file and (latest_fundamental is None or file > latest_fundamental):
                                latest_fundamental = file
                            elif 'research' in file and (latest_research is None or file > latest_research):
                                latest_research = file
                            elif 'report' in file and (latest_report is None or file > latest_report):
                                latest_report = file
                        
                        # Load the relevant analysis data
                        analysis_data = {}
                        
                        if latest_technical and ('support' in question.lower() or 'resistance' in question.lower() or 'technical' in question.lower()):
                            with open(latest_technical, 'r') as f:
                                analysis_data['technical'] = json.load(f)
                        
                        if latest_fundamental and ('fundamental' in question.lower() or 'p/e' in question.lower() or 'ratio' in question.lower()):
                            with open(latest_fundamental, 'r') as f:
                                analysis_data['fundamental'] = json.load(f)
                        
                        if latest_research and ('research' in question.lower() or 'sentiment' in question.lower() or 'news' in question.lower()):
                            with open(latest_research, 'r') as f:
                                analysis_data['research'] = json.load(f)
                        
                        if latest_report:
                            with open(latest_report, 'r') as f:
                                analysis_data['report'] = json.load(f)
                        
                        # Generate response based on question
                        if 'support' in question.lower() and 'level' in question.lower():
                            if 'technical' in analysis_data and 'techsummary' in analysis_data['technical']:
                                tech_data = analysis_data['technical']['techsummary']
                                st.markdown(f"<p>Based on the technical analysis of {selected_stock}, I found information about support levels in our data:</p>", unsafe_allow_html=True)
                                
                                # Extract just the support level part
                                if 'Support Levels' in tech_data:
                                    support_section = tech_data.split('Support Levels:')[1].split('\n')[0]
                                    st.markdown(f"<div class='card-content'><p><strong>Support Levels:</strong> {support_section}</p></div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div class='card-content'>{tech_data}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p>I don't have specific information about support levels for {selected_stock} in the current analysis.</p>", unsafe_allow_html=True)
                        
                        elif 'p/e' in question.lower() or ('ratio' in question.lower() and 'p' in question.lower()):
                            if 'fundamental' in analysis_data and 'summary' in analysis_data['fundamental']:
                                fund_data = analysis_data['fundamental']['summary']
                                st.markdown(f"<p>For {selected_stock}'s P/E ratio, here's what I found in our fundamental analysis:</p>", unsafe_allow_html=True)
                                
                                if 'P/E Ratio' in fund_data:
                                    pe_section = fund_data.split('P/E Ratio:')[1].split('\n')[0]
                                    st.markdown(f"<div class='card-content'><p><strong>P/E Ratio:</strong> {pe_section}</p></div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div class='card-content'>{fund_data}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p>I don't have P/E ratio information for {selected_stock} in the current analysis.</p>", unsafe_allow_html=True)
                        
                        elif 'sentiment' in question.lower():
                            if 'research' in analysis_data and 'researchreport' in analysis_data['research']:
                                research_data = analysis_data['research']['researchreport']
                                st.markdown(f"<p>Regarding sentiment analysis for {selected_stock}, here's what I found:</p>", unsafe_allow_html=True)
                                
                                if 'Sentiment' in research_data:
                                    sentiment_section = research_data.split('Sentiment')[1].split('##')[0]
                                    st.markdown(f"<div class='card-content'><p>{sentiment_section}</p></div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<p>I couldn't find specific sentiment analysis in our data.</p>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p>I don't have sentiment analysis for {selected_stock} in the current data.</p>", unsafe_allow_html=True)
                        
                        else:
                            # General question - look at the report
                            if 'report' in analysis_data and 'report' in analysis_data['report']:
                                report_data = analysis_data['report']['report']
                                st.markdown(f"<p>For your question about {selected_stock}, here's relevant information from our analysis:</p>", unsafe_allow_html=True)
                                
                                # Try to find relevant sections based on keywords
                                keywords = question.lower().split()
                                relevant_sections = []
                                
                                sections = report_data.split('##')
                                for section in sections:
                                    if any(keyword in section.lower() for keyword in keywords if len(keyword) > 3):
                                        relevant_sections.append(section)
                                
                                if relevant_sections:
                                    for section in relevant_sections:
                                        st.markdown(f"<div class='card-content'><p>{section}</p></div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<p>I couldn't find information directly related to your question. Here's a summary of our analysis:</p>", unsafe_allow_html=True)
                                    
                                    # Show the executive summary if available
                                    if 'Executive Summary' in report_data:
                                        summary_section = report_data.split('Executive Summary')[1].split('##')[0]
                                        st.markdown(f"<div class='card-content'><p>{summary_section}</p></div>", unsafe_allow_html=True)
                                    else:
                                        st.markdown("<p>Please try asking a more specific question about technical analysis, fundamentals, or investment outlook.</p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p>I don't have enough information to answer this question. Please try a different question related to technical analysis, fundamentals, or investment outlook.</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="error-box">
                                <h4 style="margin-top: 0; color: #ef4444;">No Analysis Found</h4>
                                <p>No analysis files found for {selected_stock}. Please run an analysis first.</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                    <div class="error-box">
                        <h4 style="margin-top: 0; color: #ef4444;">Error</h4>
                        <p>Error retrieving analysis: {str(e)}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="card">
                <div class="card-title">No Analyzed Stocks Available</div>
                <div class="card-content">
                    <p>No stock analyses are available yet. Please analyze stocks in the Stock Analysis section first.</p>
                    <p>Once you've analyzed a stock, you can return here to ask questions about it.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

# 
# Fetch Stock Data with better error handling
def get_stock_data(stock_symbol, period='1y'):
    try:
        data = yf.download(stock_symbol, period=period)
        if data.empty:
            return None
        # Clean NaN values
        return data.ffill().bfill()  # Forward then backward fill
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Plot Stock Chart with enhanced styling
def plot_stock_chart(stock_data, indicators):
    # Ensure data is not empty
    if stock_data.empty:
        st.error("Stock data is empty. Cannot create chart.")
        return None

    # Create subplots: Candlestick and Volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price",
            increasing_line_color='#10b981',  # Green
            decreasing_line_color='#ef4444',  # Red
            increasing_fillcolor='rgba(16, 185, 129, 0.2)',  # Light green
            decreasing_fillcolor='rgba(239, 68, 68, 0.2)'  # Light red
        ),
        row=1, col=1
    )

    # Add indicators
    if 'Moving Averages' in indicators:
        # Calculate simple moving averages
        stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
        
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['SMA50'], name="50-day SMA", 
                      line=dict(color='#3b82f6', width=1.5)),  # Blue
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['SMA200'], name="200-day SMA", 
                      line=dict(color='#f59e0b', width=1.5)),  # Orange
            row=1, col=1
        )
    
    if 'RSI' in indicators:
        # Calculate RSI (14 periods)
        delta = stock_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Add RSI subplot
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['RSI'], name="RSI (14)", 
                      line=dict(color='#8b5cf6', width=1.5)),  # Purple
            row=1, col=1
        )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line=dict(color='rgba(239, 68, 68, 0.5)', width=1, dash='dash'), row=1)
        fig.add_hline(y=30, line=dict(color='rgba(16, 185, 129, 0.5)', width=1, dash='dash'), row=1)
        
    if 'MACD' in indicators:
        # Calculate MACD
        stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
        stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Add MACD subplot
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MACD'], name="MACD", 
                      line=dict(color='#3b82f6', width=1.5)),  # Blue
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['Signal'], name="Signal", 
                      line=dict(color='#ef4444', width=1.5)),  # Red
            row=1, col=1
        )

    # Volume chart - FIXED VERSION
    if 'Volume' in indicators:
        # Create a list to store colors
        colors = []
        
        # Calculate daily price changes using numpy for vectorized operations
        closes = stock_data['Close'].values
        price_change = np.diff(closes)
        # Add an initial color (doesn't matter which one since it's just one bar)
        colors.append('rgba(16, 185, 129, 0.7)')  # Green for first bar
        
        # Color based on price change (up or down)
        for change in price_change:
            if change >= 0:
                colors.append('rgba(16, 185, 129, 0.7)')  # Green for positive change
            else:
                colors.append('rgba(239, 68, 68, 0.7)')  # Red for negative change
        
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name="Volume",
                marker=dict(color=colors)
            ),
            row=2, col=1
        )

    # Update layout with enhanced styling and 3D depth
    fig.update_layout(
        height=600,
        title={
            'text': f"{stock_data.index[-1].strftime('%Y-%m-%d')} - Stock Analysis",
            'font': {'size': 24, 'color': '#1e3a8a', 'family': 'Arial, sans-serif', 'weight': 'bold'},
            'y': 0.97,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend={
            'orientation': "h",
            'yanchor': "bottom",
            'y': 1.02,
            'xanchor': "right",
            'x': 1,
            'font': {'size': 12, 'family': 'Arial, sans-serif'},
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': '#e2e8f0',
            'borderwidth': 1
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': "Arial, sans-serif", 'size': 12, 'color': '#1e293b'},
        margin={'l': 40, 'r': 40, 't': 80, 'b': 40},
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        shapes=[
            # Add subtle gradient background for price chart
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0.3,
                x1=1,
                y1=1,
                fillcolor="rgba(248, 250, 252, 0.5)",
                opacity=0.5,
                line_width=0,
                layer="below"
            ),
            # Add subtle gradient background for volume chart
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=0.3,
                fillcolor="rgba(243, 244, 246, 0.5)",
                opacity=0.5,
                line_width=0,
                layer="below"
            )
        ]
    )
    
    # Grid lines and axes styling with depth
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(226, 232, 240, 0.7)',
        zerolinecolor='rgba(203, 213, 225, 1)',
        zerolinewidth=1,
        title={
            'text': "Price ($)",
            'font': {'size': 14, 'family': 'Arial, sans-serif', 'color': '#1e3a8a'}
        },
        tickfont={'size': 12, 'family': 'Arial, sans-serif'},
        tickprefix="$",
        ticksuffix="",
        showspikes=True,
        spikecolor='rgba(203, 213, 225, 0.5)',
        spikethickness=1,
        spikedash='solid',
        spikemode='across',
        row=1, col=1
    )# Update volume axis styling
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(226, 232, 240, 0.7)',
        zerolinecolor='rgba(203, 213, 225, 1)',
        zerolinewidth=1,
        title={
            'text': "Volume",
            'font': {'size': 14, 'family': 'Arial, sans-serif', 'color': '#1e3a8a'}
        },
        tickfont={'size': 12, 'family': 'Arial, sans-serif'},
        showspikes=True,
        spikecolor='rgba(203, 213, 225, 0.5)',
        spikethickness=1,
        spikedash='solid',
        spikemode='across',
        row=2, col=1
    )
    
    # Update x-axis styling
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=0.5, 
        gridcolor='rgba(226, 232, 240, 0.7)',
        zerolinecolor='rgba(203, 213, 225, 1)',
        zerolinewidth=1,
        tickfont={'size': 12, 'family': 'Arial, sans-serif'},
        showspikes=True,
        spikecolor='rgba(203, 213, 225, 0.5)',
        spikethickness=1,
        spikedash='solid',
        spikemode='across',
        row=2, col=1
    )

    # Add watermark (optional)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        text="FinWisely",
        showarrow=False,
        font=dict(
            family="Arial, sans-serif",
            size=40,
            color="rgba(226, 232, 240, 0.2)"
        ),
        opacity=0.5,
        textangle=25
    )

    return fig

def perform_crew_analysis(stock_symbol):
    with st.spinner("🧠 AI Agents are analyzing the stock... This process may take a few minutes."):
        try:
            analysis_result = run_analysis(stock_symbol)
            
            # Ensure we can extract the report content regardless of its structure
            if analysis_result and isinstance(analysis_result, dict):
                # Create properly formatted output
                formatted_result = {}
                
                # Handle different possible report structures
                if 'report' in analysis_result:
                    report_content = analysis_result['report']
                    
                    # Extract actual report text based on different possible formats
                    if isinstance(report_content, str):
                        formatted_result['report'] = report_content
                    elif hasattr(report_content, 'report'):
                        formatted_result['report'] = report_content.report
                    elif isinstance(report_content, dict) and 'report' in report_content:
                        formatted_result['report'] = report_content['report']
                    else:
                        formatted_result['report'] = str(report_content)
                
                # Copy other result sections
                for key in analysis_result:
                    if key != 'report':
                        formatted_result[key] = analysis_result[key]
                
                return formatted_result
            return analysis_result
            
        except Exception as e:
            st.markdown(f"""
                <div class="error-box">
                    <h4 style="margin-top: 0; color: #ef4444;">⚠️ Analysis Failed</h4>
                    <p>Unable to perform AI analysis: {str(e)}</p>
                    <p>Please try again or select a different stock symbol.</p>
                </div>
            """, unsafe_allow_html=True)
            st.error(traceback.format_exc())
            return None

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error(traceback.format_exc())