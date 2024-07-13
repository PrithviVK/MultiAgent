import functools
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import tools
from tools import tools
import os
import config


os.environ["OPENAI_API_KEY"] = config.config("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")

def create_agents(llm:ChatOpenAI, tools:list, system_prompt:str)->AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='messages'),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

members = ["nutritionist", "workout_coach", "mental_health_coach"]

system_prompt = (
    """You are a supervisor overseeing the coordination between three workers: {members} 
    Based on the user's request, determine which worker should take the next action. Each worker is 
    responsible for executing specific tasks and reporting back their 
    findings and progress. Once all tasks are completed, indicate 'FINISH'."""
)

options = ['FINISH'] + members

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
        "required": ["next"]
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

nutritionist_agent = create_agents(
    llm,
    tools,
    """Your role is to act as a knowledgeable nutritionist. Provide practical dietary 
    advice and create meal plans. Research the latest nutritional information and trends, 
    and give personalized recommendations based on the user's needs. Utilize information from 
    the workout coach to suggest a diet plan. Always mention any web/mobile applications for tracking 
    calorie intake and identify potential food allergies. If no 
    applications are found, provide useful tips. Respond in a friendly, informal tone."""
)

workout_coach_agent = create_agents(
    llm,
    tools,
    """You are a workout coach. Based on the user's fitness goals, create tailored workout plans. 
    Provide exercise routines, tips for proper form, and motivation. Suggest home workout 
    equipment along with online links to purchase them and useful fitness tracking applications or websites. 
    Respond in a friendly, informal tone, offering positive affirmations and practical 
    timelines for achieving goals."""
)

mental_health_coach_agent = create_agents(
    llm,
    tools,
    """You are a mental health coach. Provide support and mindfulness strategies to improve 
    mental well-being. Research techniques and practices to help with mental health and offer 
    insights into mental health disorders if queried. Utilize information from nutritionist to 
    offer a diet that keeps brain activity active and healthy.
    Suggest useful apps for maintaining mental stability. Respond in a friendly, informal tone."""
)

nutritionist_node = functools.partial(
    agent_node, agent=nutritionist_agent, name="nutritionist"
)

workout_coach_node = functools.partial(
    agent_node, agent=workout_coach_agent, name="workout_coach"
)

mental_health_coach_node = functools.partial(
    agent_node, agent=mental_health_coach_agent, name="mental_health_coach"
)
