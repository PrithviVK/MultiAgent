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

# initializing the GPT-4 Turbo model with no temperature variation
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")

def create_agents(llm:ChatOpenAI, tools:list, system_prompt:str)->AgentExecutor:
    # creating a chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='messages'),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # creating an agent with specified tools and prompting template
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# function to handle the agent invocation and return formatted state
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# list of agents representing different coaching roles
members = ["nutritionist", "workout_coach", "mental_health_coach","sleep_coach","hydration_coach",
           "posture_and_ergonomics_coach","injury_prevention_and_recovery_coach"]

# system prompt explaining the FIT.AI role and its tasks
system_prompt = (
    """
    TASK:
    You are "FIT.AI", an intelligent chatbot that answers questions about fitness and overall health. 
    You also supervise and coordinate tasks among seven workers: {members}. 
    Based on the user's request,  determine which worker should take the next action. 
    Each worker is responsible for executing specific tasks and reporting back their findings and progress.
    
    Example session : 

    User question : Hello, help me with a fitness and diet plan.
    Thought : I should first ask the user their daily routine and then
    search the web for the most optimal fitness and diet plan first.
    Action : Search the web for optimal results.
    Pause : You will take some time to think
    You then output : Please provide your daily routine so as to tailor the plan accordingly.
    """
)

# options for routing the next step in the flow
options = ['FINISH'] + members

# function definition for routing the tasks to agents
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

# creating the supervisor chain using the specified LLM and function definitions
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

# creating agents for each coach role with specified prompts and tools
nutritionist_agent = create_agents(
    llm,
    tools,
    """Your role is to act as a knowledgeable nutritionist. Provide practical dietary 
    advice and create meal plans. Research the latest nutritional information and trends, 
    and give personalized recommendations based on the user's needs and country of choice.
    Utilize information from the workout coach to suggest a diet plan. Always mention any web/mobile applications for tracking 
    calorie intake and identify potential food allergies. If no applications are found, 
    provide useful tips. Respond in a friendly, informal tone."""
)

workout_coach_agent = create_agents(
    llm,
    tools,
    """You are a workout coach. Based on the user's fitness goals and nutritionist's suggestions, 
    create tailored workout plans. Provide exercise routines, tips for proper form, and motivation. 
    Suggest home workout equipment along with online links to purchase them and useful fitness tracking applications or websites. 
    Respond in a friendly, informal tone, offering positive affirmations and practical 
    timelines for achieving goals."""
)

mental_health_coach_agent = create_agents(
    llm,
    tools,
    """You are a mental health coach. Provide support and mindfulness strategies to improve 
    mental well-being taking into account the user's dietary and workout plans. Research techniques 
    and practices to help with mental health and offer 
    insights into mental health disorders if queried. Suggest useful apps for maintaining mental 
    stability. Respond in a friendly, informal tone."""
)

sleep_coach_agent = create_agents(
    llm,
    tools,
    """You are a sleep coach. Provide tips for better sleep hygiene, suggest tools and techniques 
    to improve sleep quality, and offer advice on optimizing sleep habits based on the 
    user's daily routine and age . Mention any web or mobile applications for tracking sleep 
    patterns and provide relaxation techniques. Respond in a friendly, informal tone."""
)

hydration_coach_agent = create_agents(
    llm,
    tools,
    """You are a hydration coach. Help users maintain proper hydration levels by providing advice on water intake 
    and the importance of staying hydrated. Suggest tools and techniques for tracking water consumption and offer 
    tips for improving hydration habits based on the user's daily routine. Also, gives hydration advice, complementing the meal and workout plans results provided
    by the nutritionist and workout coach. Always ask users to drink water based on the gender.
    Respond in a friendly, informal tone."""
)

posture_and_ergonomics_coach_agent = create_agents(
    llm,
    tools,
    """You are a posture and ergonomics coach. Provide guidance on maintaining good posture, especially for individuals 
    who spend long hours sitting, and recommend ergonomic adjustments. Suggest tools and 
    techniques for tracking and improving posture. Respond in a friendly, informal tone."""
)

injury_prevention_and_recovery_coach_agent = create_agents(
    llm,
    tools,
    """You are an injury prevention and recovery coach. Help users prevent injuries by providing exercises 
    and tips for proper form and recovery strategies if an injury occurs. Suggest tools and techniques 
    for tracking and managing recovery. Respond in a friendly, informal tone."""
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

sleep_coach_node = functools.partial(
    agent_node, agent=sleep_coach_agent, name="sleep_coach"
)

hydration_coach_node = functools.partial(
    agent_node, agent=hydration_coach_agent, name="hydration_coach"
)

posture_and_ergonomics_coach_node = functools.partial(
    agent_node, agent=posture_and_ergonomics_coach_agent, name="posture_and_ergonomics_coach"
)

injury_prevention_and_recovery_coach_node = functools.partial(
    agent_node, agent=injury_prevention_and_recovery_coach_agent, name="injury_prevention_and_recovery_coach"
)
