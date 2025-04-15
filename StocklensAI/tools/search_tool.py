import json
import os
import requests
from crewai.tools import BaseTool


class SearchInternetTool(BaseTool):
    """
    A tool for searching the internet about a given topic.
    """

    def __init__(self):
        super().__init__(
            name="search_internet",
            description="Search the internet for information about a given topic and return relevant results."
        )

    def _run(self, query: str) -> str:
        """
        Perform a search query on the internet using the Serper.dev API.

        Args:
            query (str): The search query.

        Returns:
            str: The formatted search results.
        """
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.getenv('SERPER_API_KEY'),
            'content-type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch search results. Status code: {response.status_code}")

        results = response.json().get('organic', [])
        formatted_results = []

        for result in results[:top_result_to_return]:
            try:
                formatted_results.append('\n'.join([
                    f"Title: {result['title']}",
                    f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}",
                    "\n-----------------"
                ]))
            except KeyError:
                continue

        return '\n'.join(formatted_results)


class SearchNewsTool(BaseTool):
    """
    A tool for searching news about a given topic.
    """

    def __init__(self):
        super().__init__(
            name="search_news",
            description="Search for news about a company, stock, or other topics and return relevant results."
        )

    def _run(self, query: str) -> str:
        """
        Perform a news search query using the Serper.dev API.

        Args:
            query (str): The news query.

        Returns:
            str: The formatted news results.
        """
        top_result_to_return = 4
        url = "https://google.serper.dev/news"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.getenv('SERPER_API_KEY'),
            'content-type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch news results. Status code: {response.status_code}")

        results = response.json().get('news', [])
        formatted_results = []

        for result in results[:top_result_to_return]:
            try:
                formatted_results.append('\n'.join([
                    f"Title: {result['title']}",
                    f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}",
                    "\n-----------------"
                ]))
            except KeyError:
                continue

        return '\n'.join(formatted_results)
# if __name__ == "__main__":
#     search_tool = SearchInternetTool()
#     news_tool = SearchNewsTool()

#     try:
#         internet_results = search_tool.run(query="OpenAI ChatGPT updates")
#         print("Internet Search Results:\n", internet_results)

#         news_results = news_tool.run(query="OpenAI ChatGPT news")
#         print("\nNews Search Results:\n", news_results)
#     except Exception as e:
#         print(f"An error occurred: {e}")