import functools
import operator
import requests
import os
from bs4 import BeautifulSoup
# from duckduckgo_search import DDGS
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END,START
# from langchain.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain.tools.tavily_search import TavilySearchResults
import getpass
# from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from tavily import TavilyClient
import gradio as gr
from decouple import config

#Set env variables
os.environ["OPENAI_API_KEY"]=config("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"]=config("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# os.environ["TAVILY_API_KEY"]=getpass.getpass("Enter your Tavily API key: ")
# tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# os.environ["TAVILY_API_KEY"]=getpass.getpass() 

# tavily_tool = TavilySearchResults(max_results=5)
# Instantiate the TavilySearchResults class with your Tavily API key
# tavily_tool = TavilySearchResults(max_results=5, api_key=os.getenv("TAVILY_API_KEY"))
#Intialize model
llm= ChatOpenAI(temperature=0,model="gpt-4-turbo-preview")

#Define custom tools to use
@tool("process_search_tool",return_direct=False)
def process_search_tool(url:str)->str:
    #function doc string
    """Used to process content found on the internet."""
    try:
        response = requests.get(url=url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        return f"Error processing the URL: {e}"
    # response=requests.get(url=url)
    # # response = tavily_tool.search(query=url, topic="news", max_results=5)
    # soup=BeautifulSoup(response.content,"html.parser")
    # return soup.get_text()

@tool("internet_search_tool",return_direct=False)
def internet_search_tool(query:str)->str:
    """Search user query on the internet using TavilyAPI."""
    try:
        response = tavily_client.search(query=query, max_results=5)#["results"]
        return response if response else "No results found"
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {e}"
    # with DDGS() as ddgs: #context manager 3(DDGS class to use DuckDuckGo search library)
        # results=[r for r in ddgs.text(query,max_results=5)]
    # results = tavily_client.search(query=query,max_results=5)
    # return results if results else "No results found"


# search = TavilySearchAPIWrapper()
# tavily_tool = TavilySearchResults(api_wrapper=search)
# tavily=TavilySearchResults(max_results=2)
# def search_tavily(self, query: str):
#         results = tavily_client.search(query=query, topic="news", max_results=10)#include_images=True
#         sources = results["results"]
#         return sources

# tool = TavilySearchResults()
# TavilySearchResults(max_results=1)
# tavily_tool = TavilySearchResults(max_results=1)
tools=[internet_search_tool,process_search_tool]

#function to create an agent 
def create_agents(llm:ChatOpenAI, tools:list, system_prompt:str)->AgentExecutor:
    prompt=ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='messages'),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        #here scratchpad is used for the agents to write down their thoughts
    ])

    agent=create_openai_tools_agent(llm, tools, prompt)
    executor=AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state,agent,name):#state of agent, the agent itself, name of the agent(node)
    result=agent.invoke(state)# to get state of the node
    #return a dictionary to have a key of messages  
    return {"messages":[HumanMessage(content=result["output"], name=name)]}#here name is name 
    # of human message
    #so basically the output of the node is in a human readable format

#List of agents
# here we create members(agents) of our crew(agents to perform various tasks)

members=["nutritionist", "workout_coach", "mental_health_coach"]# list of nodes in the graph
system_prompt=(
    """You are a supervisor overseeing the coordination between three workers:{members} 
    Based on the user's request, determine which worker should take the next action. Each worker is 
    responsible for executing specific tasks and reporting back their 
    findings and progress. Once all tasks are completed, indicate 'FINISH'."""
    
    # "As a supervisor, your role is to oversee the insight between these"
    # "workers:{members}. Based on the user's request, "
    # "determine which worker should take the next action"
    # "Each worker is responsible for executing a specific task and" 
    # "reporting back their findings and progress."
    # "Once all tasks are completed, indicate 'FINISH'."
)

options= ['FINISH']+ members

# a routeSchema to determine which agent runs next (to check which route should be taken)
function_def={
    "name":"route",
    "description":"Select the next role.",
    "parameters": {
        "title":"routeSchema",
        "type":"object",
        "properties":{"next":{"title":"Next","anyOf":[{"enum":options}]}},
        "required":["next"]
        }    
    }

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

#we then create a news_correspondent agent 
nutritionist_agent=create_agents(
    llm,
    tools,
    """Your role is to act as a knowledgeable nutritionist. Provide practical dietary 
    advice and create meal plans. Research the latest nutritional information and trends, 
    and give personalized recommendations based on the user's needs. Utilize information from 
    the workout coach to suggest a diet plan. Always mention any web/mobile applications for tracking 
    calorie intake and identify potential food allergies. If no 
    applications are found, provide useful tips. Respond in a friendly, informal tone.
    """
    #Your primary role is to function as an intelligent news research assistant, adept at scouring 
    # the internet for the latest and most relevant trending stories across various sectors like politics, technology, 
    # health, culture, and global events. You possess the capability to access a wide range of online news sources, 
    # blogs, and social media platforms to gather real-time information.
)

#create nodes for seperate agents
nutritionist_node=functools.partial(
    agent_node,agent=nutritionist_agent,name="nutritionist"
)

workout_coach_agent=create_agents(
    llm,
    tools,
    """You are a workout coach. Based on the user's fitness goals, create tailored workout plans. 
    Provide exercise routines, tips for proper form, and motivation. Suggest home workout 
    equipment along with online links to purchase them and useful fitness tracking applications or websites. 
    Respond in a friendly, informal tone, offering positive affirmations and practical 
    timelines for achieving goals.
    Always mention all exercises are to be perfomed with proper form and posture.
    """
    # You are a news editor. Do step by step approach. 
    #     Based on the provided content first identify the list of topics,
    #     then search internet for each topic one by one
    #     and finally find insights for each topic one by one that can aid you 
    #     in writting a useful news edition for AI-generated-news corp.
    #     Include the insights and sources in the final response
)

workout_coach_node=functools.partial(
    agent_node,agent=workout_coach_agent,name="workout_coach"
)


mental_health_coach_agent=create_agents(
    llm,
    tools,
    """You are a mental health coach. Provide support and mindfulness strategies to improve 
    mental well-being. Research techniques and practices to help with mental health and offer 
    insights into mental health disorders if queried. 
    Suggest useful apps for maintaining mental stability. Respond in a friendly, informal tone."""
    # """You are an ads writter for AI-generated-news corp. Given the publication generated by the
    # news editor, your work if to write ads that relate to that content. Use the internet 
    # to search for content to write ads based off on. Here is a description of your task:
    # To craft compelling and relevant advertisements for 'AI-generated-news' publication, complementing 
    # the content written by the news editor.
    # Contextual Ad Placement: Analyze the final report content from the news editor in-depth to 
    # identify key themes, topics, and reader interests. Place ads that are contextually 
    # relevant to these findings, thereby increasing potential customer engagement.
    # Advanced Image Sourcing and Curation: Employ sophisticated web search algorithms to source 
    # high-quality, relevant images for each ad. 
    # Ensure these images complement the ad content and are aligned with the publication's 
    # aesthetic standards. 
    # Ad-Content Synchronization: Seamlessly integrate advertisements with the 
    # report, ensuring they enhance rather than disrupt the reader's experience. Ads should feel like 
    # a natural extension of the report, offering value to the reader.
    # Reference and Attribution Management: For each image sourced, automatically 
    # generate and include appropriate references and attributions, ensuring compliance 
    # with copyright laws and ethical standards."""
)
mental_health_coach_node=functools.partial(
    agent_node,agent=mental_health_coach_agent,name="mental_health_coach"
)

#now we write a class to keep track of agent states and how well each agent interacts with each other
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],operator.add]
    next: str
# class AgentState(TypedDict):
#     messages:Annotated(Sequence[BaseMessage],operator.add) 
#     next:str

#now we create a stateful graph or also called workflow
workflow=StateGraph(AgentState) #instantiating a workflow 

#we now add nodes to the workflow
workflow.add_node("supervisor", action=supervisor_chain)
workflow.add_node("nutritionist", action=nutritionist_node)
workflow.add_node("workout_coach", action=workout_coach_node)
workflow.add_node("mental_health_coach", action=mental_health_coach_node)

# we define edges to connect all nodes together
for member in members:
    #here the workflow starts with a member and ends with the supervisor
    workflow.add_edge(start_key=member, end_key="supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
#we now add a conditional edge using a conditional map
    #in the below code it specifies that if a task is 'Finished' the supervisor will not send the 
    #task to an agent else, supervisor will continue to send the task until task is completed
    # workflow.add_conditional_edges("supervisor", lambda x:x["next"] if x["next"] in options else END, 
    #                                conditional_map)
workflow.add_conditional_edges("supervisor",lambda x:x["next"], conditional_map)
    # workflow.set_entry_point("supervisor")
workflow.add_edge(START,"supervisor")

# print(workflow.branches)
# print(workflow.edges)
# print(workflow.nodes)
# print(workflow.channels)#intermediate steps agents can take 

graph=workflow.compile()

#below is the code functionality for streaming(invoking the compiled graph)
# for s in graph.stream(
#     {
#         "messages": [
#             HumanMessage(content=
#                          """I want to improve my overall fitness. After research on the given user query
#                          Start with a meal plan from the nutritionist based, then get a workout routine from 
#                          the workout coach, and finally get some mental health tips from the mental health coach.
#                          Also,give me suggestions for a better and healtier lifestyle."""
#                         )

#         #                  """Write me a report on spaceX. After the research on spaceX,
#         #                       pass the findings to the news editor to generate the final publication.
#         #                       Once done, pass it to the ads writter to write the ads on the subject."""
#         ]
#     },
#     {"recursion_limit":150} #maximum number of steps to take in the graph
#     # the number 150 represents the number of back and forth conversations between agents
# ):
#     if "__end__" not in s:
#         print(s)
#         print("----")

# no streaming 
final_respone = graph.invoke({
    "messages": [HumanMessage(content=
                              """I want to improve my overall fitness. After research on the given user query
                              Start with a meal plan from the nutritionist based, then get a workout routine from 
                              the workout coach, and finally get some mental health tips from the mental health coach.
                              Also,give me suggestions for a better and healtier lifestyle.""")]

# """Write me a report on spaceX. After the research on spaceX,
#                               pass the findings to the news editor to generate the final publication.
#                               Once done, pass it to the ads writter to write the ads on the subject."""
}, {"recursion_limit": 150})

print(final_respone["messages"][1].content)


def run_graph(input_message):
    response=graph.invoke({
        "messages":[HumanMessage(content=input_message)]
    })
    return response['messages'][1].content

inputs = gr.components.Textbox(lines=5, placeholder="Enter your query")
outputs = gr.components.Markdown()

demo = gr.Interface(
    fn=run_graph,
    inputs=inputs,
    outputs=outputs
)

demo.launch()












































