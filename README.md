# Multiagent Fitness Application<br>
FIT.AI is an intelligent multi-agent fitness application backed by the most popular framework, __LangGraph__ utilizing the most popular Large Language Model, 
__gpt-4-turbo-preview__ and a search engine optimized for LLM’s and RAG(Retrieval Augmented Generation), __Tavily API__.

## Features
1. Crafting diet plans based on user information such as age, gender, height, weight.
2. Crafting workout plans based on user information such as age, gender, height, weight.
3. BMI Calculator. 
4. Provides tips for a better mental health and sleep patterns.
5. Provides water consumption level required for a person of different age and gender.
6. Assists with postures in the workplace.
7. Assists with injury rehabilitation.

## Architecture
The application consists of the following components:<br>
1. User
2. Front End
3. Back End
   - supervisor - agent that controls the flow of data between other sub agents
   - nutritionist - provides personal diet
   - workout coach - provides personal workouts
   - mental health coach - mental health tips
   - sleep coach - sleeping habits
   - hydration coach - daily water intake  
   - posture and ergonomics coach - daily postures in different settings
   - injury prevention and recovery coach - aids in recovery<br><br>

<img width="500" alt="image" height="300" src="https://github.com/user-attachments/assets/0cf0f809-6aea-47d8-b773-61bc5a0d361f"><br><br>

## WorkFlow
![Flowcharts](https://github.com/user-attachments/assets/ac14c0b0-912a-4657-b2ec-3698f1fb168d)



## Project Setup
1. Install poetry.
   
3. To install _poetry_ we need pipx pre-installed based on our operating system.
   If not installed please refer to https://pipx.pypa.io/stable/installation/
4. To install poetry please type in the following: `pipx install poetry`
5. Setting up our virtual environment. Type in the following command<br>
   `poetry config virtualenvs.in-project true`
6. Now, install _langchain, langgraph, langchain-openai, python-decouple, gradio, beautifulsoup4, tavily-python_ inside the downloaded **multiagent** directory using the following command:<br>
   `poetry add langchain langgraph langchain-openai python-decouple gradio beautifulsoup4 tavily-python`
7. If faced with an error try updating your python version to the latest version in the pyproject.toml file and run the above given command again.
8. Now retrieve your API keys from the following websites :<br>
For OPEN AI Key : https://platform.openai.com/api-keys<br>
For Tavily API Key : https://docs.tavily.com/docs/welcome
   
9. Once done with the installation and API key retrieval, create a `.env` file in the main multiagent directory outside of the sections and test folder and provide the following input:<br>

```
OPENAI_API_KEY= YOUR_API_KEY
TAVILY_API_KEY= YOUR_API_KEY
```

10. Navigate to the _sections_ directory present in the _multiagent_ directory and open the terminal.
11. Now we run the project in the present working directory(_sections_) using the following the following command:<br>
`poetry run python3 main.py`
12. Open link to localhost displayed on your terminal.
13. Ask away your fitness related queries.


## Usage
The bot utilizes the openai llm and tavily api to provide general fitness related user queries.
Open the folder downloaded in an IDE such as [VisualStudio code](https://code.visualstudio.com/download).
Retrive your API Keys as specified above and paste it in the config.py file<br>

```
import os
from decouple import config
# Set environment variables
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = config("TAVILY_API_KEY")
```

After following the above mentioned steps run the code on your terminal.<br>
`poetry run python3 main.py`
