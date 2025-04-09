import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from crew import run_analysis
import json
import pandas as pd
import traceback

# Custom CSS for Styling
def add_custom_css():
    st.markdown("""
        <style>
            .cleaned_dataview-container { background: white; }
            .sidebar .sidebar-content { background: white; }
            .big-font { font-size:30px !important; font-weight: bold; color: #1E3A8A; }
            .medium-font { font-size:20px !important; font-weight: bold; color: #1E3A8A; }
            .small-font { font-size:14px !important; color: #555555; }
            .analysis-card { background-color: black; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
            .analysis-title { font-size: 18px; font-weight: bold; color: #1E3A8A; margin-bottom: 10px; }
            .analysis-content { font-size: 14px; color: #FFFFFF; line-height: 1.6; }
        </style>
    """, unsafe_allow_html=True)

# Main Function
def main():
    st.set_page_config(layout="wide", page_title="FinWisely: Your AI Financial Guide using Multi-Agent Systems")
    add_custom_css()
    st.markdown('<p class="big-font">FinWisely: Your AI Financial Guide using Multi-Agent Systems</p>', unsafe_allow_html=True)
    st.markdown("""
        <div class="analysis-card">
            <p class="analysis-content">
                Welcome to FinWisely, the all-in-one AI-powered financial platform designed to revolutionize the way you navigate your financial journey. Whether you're taking your first steps toward financial independence or looking to optimize your wealth-building strategies, FinWisely empowers you with cutting-edge tools, personalized insights, and comprehensive education.
                The mission is to simplify financial decision-making, enabling you to take control of your money and achieve long-term financial success. FinWisely seamlessly integrates financial literacy education, smart budgeting tools, and AI-driven investment strategies, all tailored to meet your unique goals. 
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Choose a Section", ["Stock Analysis", "Financial Literacy", "Budgeting"])

    if options == "Stock Analysis":
        stock_analysis_section()
    elif options == "Financial Literacy":
        financial_literacy_section()
    elif options == "Budgeting":
        budgeting_section()

def stock_analysis_section():
    st.markdown('<h2 class="medium-font">AI Stock Analysis</h2>', unsafe_allow_html=True)
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
    time_period = st.sidebar.selectbox("Select Time Period", ['3mo', '6mo', '1y', '2y', '5y'])
    indicators = st.sidebar.multiselect("Select Indicators", ['Moving Averages', 'Volume', 'RSI', 'MACD'])
    analyze_button = st.sidebar.button("📊 Analyze Stock")
    
    if analyze_button:
        try:
            # Show a message to let the user know the process is starting
            with st.spinner("Fetching stock data..."):
                stock_data = get_stock_data(stock_symbol, time_period)
                
            if stock_data is not None and not stock_data.empty:
                # Display chart first to give user immediate feedback
                st.plotly_chart(plot_stock_chart(stock_data, indicators), use_container_width=True)
                
                # Proceed with agent analysis
                with st.spinner("🚀 AI Agents are analyzing the stock... This might take a few minutes."):
                    analysis = perform_crew_analysis(stock_symbol)
                
                if analysis:
                    st.success("✅ Agents have completed the stock analysis!")
                    st.markdown("""
                        <div class="analysis-card">
                            <p class="analysis-content">
                                Agents successfully analyzed the stock and provided insights across technical, fundamental, and sentiment aspects.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display analysis results
                    if 'report' in analysis:
                        st.markdown("### Analysis Report")
                        st.write(analysis['report'])
            else:
                st.error(f"Could not retrieve data for symbol {stock_symbol}. Please check if the symbol is correct.")
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.error(traceback.format_exc())
        
# Fetch Stock Data
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

# Plot Stock Chart
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
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # Add indicators
    if 'Moving Averages' in indicators:
        # Calculate simple moving averages
        stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
        
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['SMA50'], name="50-day SMA", line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['SMA200'], name="200-day SMA", line=dict(color='orange')),
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
            go.Scatter(x=stock_data.index, y=stock_data['RSI'], name="RSI (14)", line=dict(color='purple')),
            row=1, col=1
        )
        
    if 'MACD' in indicators:
        # Calculate MACD
        stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
        stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Add MACD subplot
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['MACD'], name="MACD", line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['Signal'], name="Signal", line=dict(color='red')),
            row=1, col=1
        )

    # Volume chart
    if 'Volume' in indicators:
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name="Volume",
                marker=dict(color='blue')
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=800,
        title=f"Stock Chart Analysis for {stock_data.index[-1].strftime('%Y-%m-%d')}",
        xaxis_rangeslider_visible=False,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        yaxis=dict(title="Price", gridcolor='gray'),
        yaxis2=dict(title="Volume", gridcolor='gray'),
        xaxis=dict(gridcolor='gray')
    )

    return fig


def perform_crew_analysis(stock_symbol):
    with st.spinner("🚀 Agents at work... performing comprehensive analysis!"):
        try:
            analysis_result = run_analysis(stock_symbol)
            return analysis_result
        except Exception as e:
            st.error(f"⚠️ Failed to perform AI analysis: {e}")
            st.error(traceback.format_exc())
            return None


def financial_literacy_section():
    st.markdown('<h2 class="medium-font">Financial Literacy Hub</h2>', unsafe_allow_html=True)
    st.markdown("""
        <p>Welcome to the Financial Literacy Hub! Learn the fundamentals of personal finance, including budgeting, saving, investing, and debt management.
        Select a topic below to get started:</p>
    """, unsafe_allow_html=True)

    topics = ["Budgeting Basics", "Investing 101", "Debt Management", "Retirement Planning"]
    selected_topic = st.selectbox("Choose a Topic", topics)

    if selected_topic == "Budgeting Basics":
        st.write("Budgeting Basics: Learn how to create a budget and stick to it.")
    elif selected_topic == "Investing 101":
        st.write("Investing 101: Understand the basics of investing and how to grow your wealth.")
    elif selected_topic == "Debt Management":
        st.write("Debt Management: Learn strategies to manage and reduce your debt effectively.")
    elif selected_topic == "Retirement Planning":
        st.write("Retirement Planning: Plan your financial future for a secure retirement.")

def budgeting_section():
    st.markdown('<h2 class="medium-font">Smart Budgeting Tool</h2>', unsafe_allow_html=True)
    income = st.number_input("Enter your Monthly Income ($):", min_value=0.0, step=100.0)
    expenses = st.number_input("Enter your Monthly Expenses ($):", min_value=0.0, step=100.0)

    if st.button("Calculate Savings"):
        savings = income - expenses
        if savings < 0:
            st.error(f"Your monthly budget is in deficit: ${-savings:.2f}")
        else:
            st.success(f"Your monthly savings: ${savings:.2f}")
        st.write("Tip: Aim to save at least 20% of your income each month for long-term financial stability.")

if __name__ == "__main__":
    main()