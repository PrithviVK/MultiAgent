import functools
import operator
import requests
import os
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain. tools import tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence
import gradio
from decouple import config

#Set env variables
os.environ["OPENAI_API_KEY"]=config("OPENAI_API_KEY")

#Intialize model
llm= ChatOpenAI(model="gpt-3.5-turbo-0125")

#Define custom tools to use
@tool("process_search_tool",return_direct=False)
def process_search_tool(url:str)->str:
    #function doc string
    """Used to process content found on the internet using DuckDuckGo search."""
    response=requests.get(url=url)
    soup=BeautifulSoup(response.content,"html.parser")
    return soup.get_text()



















