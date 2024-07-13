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
                              the workout coach, and finally get some mental health tips from the mental health coach.
                              Also,give me suggestions for a better and healtier lifestyle.""")]

# """Write me a report on spaceX. After the research on spaceX,
#                               pass the findings to the news editor to generate the final publication.
#                               Once done, pass it to the ads writter to write the ads on the subject."""
}, {"recursion_limit": 150})

print(final_respone["messages"][1].content)

def run_graph(input_message,history):
    response = graph.invoke({
        "messages": [HumanMessage(content=input_message)]
    })
    return response['messages'][1].content

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])


bot = gr.Chatbot(render=False,placeholder="<strong>Your Personal AI Assistant</strong><br>Ask Me Anything",
                           show_copy_button=True,
                           layout="bubble",
                           likeable=True)


# with gr.Blocks(title="Fitness Bot") as demo:
#     chatbot=gr.Chatbot(run_graph,placeholder="<strong>Your Personal AI Assistant</strong><br>Ask Me Anything",
#                            show_copy_button=True,
#                            layout="bubble",
#                            )
#     chatbot.like(vote, None, None,show_progress="full")

    # gr.ChatInterface(
    #     fn=run_graph,
    #     clear_btn="🗑️ Clear",
    #     theme="soft",
    #     undo_btn="Delete Previous",
    #     cache_examples=True,
    #     autofocus=True,
    #     textbox=gr.Textbox(placeholder="Ask away any fitness relates questions", scale=7),
    #     stop_btn="Stop",
    #     show_progress="full",
    #     description="<strong>An intelligent assistant for fitness, diet and mental health guidance.<strong>",
    #     js="custom.js",
    #     chatbot=bot,
    #     # chatbot=gr.Chatbot(render=False,placeholder="<strong>Your Personal AI Assistant</strong><br>Ask Me Anything",
    #     #                    show_copy_button=True,
    #     #                    layout="bubble"), 
    # )

demo = gr.ChatInterface(
    fn=run_graph,
    clear_btn="🗑️ Clear",
    theme="soft",
    undo_btn="Delete Previous",
    cache_examples=True,
    autofocus=True,
    textbox=gr.Textbox(placeholder="Ask away any fitness related questions", scale=7),
    # chatbot=gr.Chatbot(placeholder="<strong>Your Personal AI Assistant</strong><br>Ask Me Anything"),
    stop_btn="Stop",
    show_progress="full",
    description="<strong>An intelligent assistant for fitness, diet and mental health guidance.<strong>",
    js="custom.js",
    chatbot=bot,
)

demo.launch()
