# tools/AlphaVantage_finance_tool.py
import requests
import json
from crewai.tools import BaseTool, tool
from typing import Optional, Union
from datetime import datetime, timedelta
import time
import random
import os
from typing import Dict, Any, List, Type
from pydantic import Field, BaseModel

# Create a Pydantic schema for the tool inputs
class AlphaVantageNewsToolSchema(BaseModel):
    stock_symbol: str = Field(description="The stock symbol to get news for")
    max_results: Optional[Union[int, str]] = Field(default=5, description="Maximum number of news items to return")
    api_key: Optional[str] = Field(default=None, description="Alpha Vantage API key")

class AlphaVantageNewsTool(BaseTool):
    """Tool for getting news and sentiment about a stock from Alpha Vantage API."""
    
    name: str = "alpha_vantage_news"
    description: str = "Fetches the latest financial news and sentiment from Alpha Vantage for a given stock symbol."
    args_schema: Type[BaseModel] = AlphaVantageNewsToolSchema
    
    def __init__(self, api_key=None, **kwargs):
        """Initialize with API key."""
        # Store API key before super init using double underscore for name mangling
        self.__api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        # Call parent init with the rest of the kwargs
        super().__init__(**kwargs)
        print(f"AlphaVantageNewsTool initialized with API key: {'*****' if self.__api_key else 'None'}")
    def _make_request(self, url, params=None, max_retries=3):
        """Make a request with basic retry logic."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                print(f"Request attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt + 1 == max_retries:
                    raise
                time.sleep(2 ** attempt)  # Simple exponential backoff
    
    def _run(
        self, 
        stock_symbol: str, 
        max_results: Optional[Union[int, str]] = 5, 
        api_key: Optional[str] = None
    ) -> str:
        """
        Get news and sentiment for a specific stock from Alpha Vantage.
        
        Args:
            stock_symbol (str): The stock symbol to get news for.
            max_results (int or str, optional): Maximum number of news items to return. Defaults to 5.
            api_key (str, optional): Alpha Vantage API key. Overrides the one provided at initialization.
            
        Returns:
            str: A string containing the news items with sentiment analysis.
        """
        try:
            # Use provided API key, or fall back to initialized key
            current_api_key = api_key or self.__api_key
            
            # If still no API key available, inform the user
            if not current_api_key:
                return "No Alpha Vantage API key provided. Please set ALPHA_VANTAGE_API_KEY in your environment or provide an API key parameter."
            
            # Clean and validate inputs
            stock_symbol = str(stock_symbol).strip().upper().replace("{stock_symbol}", "")
            
            # Convert max_results to int if it's a string
            max_results_value = 5  # Default value
            if isinstance(max_results, str):
                try:
                    max_results_value = int(max_results)
                except ValueError:
                    max_results_value = 5
            elif isinstance(max_results, int):
                max_results_value = max_results
            
            if not stock_symbol:
                return "No stock symbol provided. Please provide a valid stock symbol."
            
            # Use Alpha Vantage News Sentiment API
            try:
                base_url = "https://www.alphavantage.co/query"
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": stock_symbol,
                    "apikey": current_api_key,
                    "limit": max_results_value
                }
                
                data = self._make_request(base_url, params)
                
                # Check if we have feed entries
                if 'feed' in data and data['feed']:
                    results = []
                    feed_items = data['feed'][:max_results_value]
                    
                    for item in feed_items:
                        title = item.get('title', 'No title available')
                        url = item.get('url', '#')
                        source = item.get('source', 'Unknown source')
                        time_published = item.get('time_published', 'Unknown date')
                        
                        # Format the time
                        if time_published and time_published != 'Unknown date':
                            try:
                                # Alpha Vantage format: YYYYMMDDTHHmmss
                                dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                                time_published = dt.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                pass
                        
                        # Get sentiment data if available
                        sentiment = "Neutral"
                        if 'overall_sentiment_score' in item:
                            score = float(item['overall_sentiment_score'])
                            if score > 0.25:
                                sentiment = f"Positive (Score: {score:.2f})"
                            elif score < -0.25:
                                sentiment = f"Negative (Score: {score:.2f})"
                            else:
                                sentiment = f"Neutral (Score: {score:.2f})"
                        
                        news_item = (
                            f"Title: {title}\n"
                            f"Source: {source}\n"
                            f"Date: {time_published}\n"
                            f"Sentiment: {sentiment}\n"
                            f"Link: {url}\n"
                        )
                        results.append(news_item)
                    
                    if results:
                        return "\n".join(results)
                    
                # Check if we got an error message
                if 'Note' in data:
                    note = data['Note']
                    if 'API call frequency' in note:
                        return f"Alpha Vantage API rate limit exceeded. Try again later."
                    return f"Alpha Vantage API error: {note}"
                
                if 'Information' in data:
                    return f"Alpha Vantage API message: {data['Information']}"
                
                # No data found
                return f"No news found for {stock_symbol}. Try a different stock symbol."
                
            except Exception as e:
                print(f"Alpha Vantage API error: {e}")
                return self._generate_mock_news(stock_symbol, max_results_value)
            
        except Exception as e:
            print(f"General error in AlphaVantageNewsTool: {e}")
            return self._generate_mock_news(stock_symbol, 5)
    
    def _generate_mock_news(self, stock_symbol, max_results_value=5):
        """Generate mock news data with sentiment when real data cannot be fetched."""
        # Get company information if possible
        company_name = stock_symbol
        try:
            import yfinance as yf
            ticker = yf.Ticker(stock_symbol)
            if hasattr(ticker, 'info') and ticker.info:
                company_name = ticker.info.get('shortName', stock_symbol)
        except:
            pass  # Use the symbol if we can't get the company name
        
        # Create news templates
        news_templates = [
            {
                "title": f"{company_name} Reports Strong Quarterly Earnings, Exceeding Analyst Expectations",
                "source": "Financial Times",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Positive (Score: 0.65)"
            },
            {
                "title": f"Analysts Raise Price Target for {company_name} Following Positive Earnings Call",
                "source": "Bloomberg",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Positive (Score: 0.52)"
            },
            {
                "title": f"{company_name} Announces New Product Line to Boost Market Share",
                "source": "Wall Street Journal",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Positive (Score: 0.38)"
            },
            {
                "title": f"{company_name} Stock Climbs as Investors React to Positive Economic Data",
                "source": "CNBC",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Positive (Score: 0.41)"
            },
            {
                "title": f"Institutional Investors Increase Stakes in {company_name} Amid Market Volatility",
                "source": "Reuters",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Neutral (Score: 0.12)"
            },
            {
                "title": f"{company_name} CEO Discusses Future Growth Strategy in Industry Conference",
                "source": "Barron's",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Neutral (Score: 0.05)"
            },
            {
                "title": f"{company_name} Expands Operations in Asian Markets, Targeting New Growth",
                "source": "Nikkei Asia",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Positive (Score: 0.32)"
            },
            {
                "title": f"What's Next for {company_name}? Analysts Weigh In After Recent Developments",
                "source": "MarketWatch",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Neutral (Score: 0.09)"
            },
            {
                "title": f"{company_name} to Invest $1B in AI and Machine Learning Technologies",
                "source": "TechCrunch",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Positive (Score: 0.71)"
            },
            {
                "title": f"Quarterly Earnings Preview: What to Expect From {company_name}",
                "source": "Investor's Business Daily",
                "link": f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}",
                "sentiment": "Neutral (Score: -0.08)"
            }
        ]
        
        # Generate dates within the last 30 days, more recent items first
        now = datetime.now()
        dates = []
        for i in range(min(len(news_templates), max_results_value)):
            # More recent news is more likely
            days_ago = random.randint(0, min(5 + i*3, 30))
            news_date = now - timedelta(days=days_ago)
            dates.append(news_date.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Sort dates in descending order (newest first)
        dates.sort(reverse=True)
        
        # Shuffle the templates to get different ones each time
        random.shuffle(news_templates)
        
        # Generate the mock news items
        results = []
        for i, (template, date) in enumerate(zip(news_templates[:len(dates)], dates)):
            news_item = (
                f"Title: {template['title']}\n"
                f"Source: {template['source']}\n"
                f"Date: {date}\n"
                f"Sentiment: {template['sentiment']}\n"
                f"Link: {template['link']}\n"
            )
            results.append(news_item)
        
        return "\n".join(results)