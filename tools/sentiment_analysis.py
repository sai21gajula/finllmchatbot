import os
import praw
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from crewai.tools import BaseTool
from typing import List

class RedditSentimentAnalysisTool(BaseTool):
    """
    A class-based tool for performing sentiment analysis on Reddit posts about a stock symbol.
    """

    def __init__(self):
        super().__init__(
            name="reddit_sentiment_analysis",
            description="Performs sentiment analysis on Reddit posts for a given stock symbol.",
        )
        # Initialize tokenizer, model, and Reddit API client
        self._tokenizer = AutoTokenizer.from_pretrained(
            "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        )
        self._reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of a given text.
        """
        inputs = self._tokenizer(text, return_tensors="pt")
        outputs = self._model(**inputs)
        scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
        labels = ["negative", "neutral", "positive"]
        label = labels[scores.argmax()]
        return label

    def get_reddit_posts(self, subreddit_name, stock_symbol, limit=100, days=30):
        """
        Get posts from a specific subreddit containing the stock symbol within the last specified days.
        """
        subreddit = self._reddit.subreddit(subreddit_name)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        posts = []
        for post in subreddit.search(stock_symbol, sort='new', time_filter='month', limit=limit):
            post_date = datetime.utcfromtimestamp(post.created_utc)
            if start_date <= post_date <= end_date:
                posts.append(post.title)
        return posts

    def _run(self, stock_symbol: str, subreddits: List[str] = None, limit: int = 100):
        """
        Perform sentiment analysis on posts from specified subreddits about a stock symbol.
        
        Args:
            stock_symbol (str): The stock symbol to search for.
            subreddits (list): List of subreddits to search in.
            limit (int): Number of posts to fetch from each subreddit.
        
        Returns:
            dict: Counts of sentiment labels across all analyzed posts.
        """
        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            
        sentiment_counts = {'neutral': 0, 'negative': 0, 'positive': 0}
        
        for subreddit in subreddits:
            posts = self.get_reddit_posts(subreddit, stock_symbol, limit)
            for post in posts:
                sentiment = self.analyze_sentiment(post)
                sentiment_counts[sentiment] += 1

        return sentiment_counts