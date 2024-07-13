import os
from decouple import config

# Set environment variables
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = config("TAVILY_API_KEY")
