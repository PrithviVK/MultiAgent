import gradio as gr
from langchain_core.messages import HumanMessage
from config import *
from tools import tools
from agents import *
from workflow import create_workflow

graph = create_workflow()

final_respone = graph.invoke({
    "messages": [HumanMessage(content=
                              """I want to improve my overall fitness. After research on the given user query
                              Start with a meal plan from the nutritionist based, then get a workout routine from 
                              the workout coach. Subsequently, get some mental health tips from the mental health coach
                              based on the dietary and workout plans. Provide hydration tips to users complementing the 
                              meal and workout plans. Finally, based on the workout plan provided by the workout coach suggest tips
                              for avoiding and managing injuries during workouts
                              Also,give me suggestions for a better and healtier lifestyle.
                              Format the answer in well written manner.""")]

}, {"recursion_limit": 150})

print(final_respone["messages"][1].content)

def run_graph(input_message,history):
    response = graph.invoke({
        "messages": [HumanMessage(content=input_message)]
    })
    return response['messages'][1].content


bot = gr.Chatbot(render=False,placeholder="<strong>Your Personal Assistant</strong><br>Ask Me Anything",
                           show_copy_button=True,
                           layout="bubble",
                           container=True,
                           label="FIT.AI",
                           show_label=True,
                           avatar_images=("images/user.png","images/bot.png"),
                           likeable=True)


demo = gr.ChatInterface(
    fn=run_graph,
    clear_btn="🗑️ Clear",
    theme="soft",
    undo_btn="Delete Previous",
    autofocus=True,
    textbox=gr.Textbox(placeholder="Ask away any fitness related questions", scale=7),
    stop_btn="Stop",
    show_progress="full",
    description="<strong>An intelligent assistant for fitness, diet and mental health guidance.<strong>",
    js="custom.js",
    examples=["Provide health and fitness tips", "My daily Calorie intake", 
              "Better mental health","Best sleep habits","Water intake for a fully grown adult",
              "Ergonomics in the workplace","Injuries Rehabilitation"],
    chatbot=bot,
)

demo.launch()
