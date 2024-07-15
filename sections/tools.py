import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from tavily import TavilyClient
import os

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# tool for processing web content
@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Used to process content found on the internet."""
    try:
        response = requests.get(url=url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        return f"Error processing the URL: {e}"

# tool for internet searches
@tool("internet_search_tool", return_direct=False)
def internet_search_tool(query: str) -> str:
    """Search user query on the internet using TavilyAPI."""
    try:
        response = tavily_client.qna_search(query=query, max_results=5)
        return response if response else "No results found"
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {e}"

tools = [internet_search_tool, process_search_tool]
