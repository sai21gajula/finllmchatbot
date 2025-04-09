import requests
import json
from crewai.tools import BaseTool
from typing import Optional
import yfinance as yf
from bs4 import BeautifulSoup

class YahooFinanceNewsTool(BaseTool):
    """A tool for getting news about a stock from Yahoo Finance."""
    
    def __init__(self):
        super().__init__(
            name="yahoo_finance_news",
            description="Fetches the latest financial news from Yahoo Finance for a given stock symbol."
        )
    
    def _run(self, stock_symbol: str, max_results: Optional[int] = 5) -> str:
        """
        Get news for a specific stock from Yahoo Finance.
        
        Args:
            stock_symbol (str): The stock symbol to get news for.
            max_results (int, optional): Maximum number of news items to return. Defaults to 5.
            
        Returns:
            str: A string containing the news items.
        """
        try:
            # Clean the stock symbol
            stock_symbol = stock_symbol.strip().upper().replace("{stock_symbol}", "")
            
            if not stock_symbol:
                return "No stock symbol provided. Please provide a valid stock symbol."
            
            # Use yfinance to get news
            ticker = yf.Ticker(stock_symbol)
            news = ticker.news
            
            if not news:
                return f"No recent news found for {stock_symbol}."
            
            # Format the results
            results = []
            for i, item in enumerate(news[:max_results]):
                title = item.get('title', 'No title available')
                link = item.get('link', '#')
                publisher = item.get('publisher', 'Unknown publisher')
                publish_time = item.get('providerPublishTime', 'Unknown date')
                
                # Format timestamp if available
                if isinstance(publish_time, int):
                    from datetime import datetime
                    publish_time = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M:%S')
                
                news_item = f"Title: {title}\nPublisher: {publisher}\nDate: {publish_time}\nLink: {link}\n"
                results.append(news_item)
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error fetching news for {stock_symbol}: {str(e)}"