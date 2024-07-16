import gradio as gr
from langchain_core.messages import HumanMessage
from config import *
from tools import tools
from agents import *
from workflow import create_workflow

# create the workflow graph
graph = create_workflow()

# function to handle user input and run the workflow graph
def run_graph(input_message, history):
    try:
        # relevant fitness-related keywords to handle irrelevant user prompts
        relevant_keywords = [
            "workout", "training", "exercise", "cardio", "strength training", "hiit (high-intensity interval training)",
            "flexibility", "yoga", "pilates", "aerobics", "crossfit", "bodybuilding", "endurance", "running",
            "cycling", "swimming", "martial arts", "stretching", "warm-up", "cool-down", 
            "diet plan", "meal plan", "macronutrients", "micronutrients", "vitamins", "minerals", "protein",
            "carbohydrates", "fats", "calories", "calorie", "daily", "nutrition", "supplements", "hydration", "weightloss",
            "weight gain", "healthy eating","health", "fitness", "intermittent fasting", "keto diet", "vegan diet", "paleo diet",
            "mediterranean diet", "gluten-free", "low-carb", "high-protein", "bmi", "calculate", "body mass index", 'calculator'
            "mental health", "mindfulness", "meditation", "stress management", "anxiety relief", "depression",
            "positive thinking", "motivation", "self-care", "relaxation", "sleep hygiene", "therapy",
            "counseling", "cognitive-behavioral therapy (cbt)", "mood tracking", "mental", "emotional well-being",
            "healthy lifestyle", "fitness goals", "health routines", "daily habits", "ergonomics",
            "posture", "work-life balance", "workplace", "habit tracking", "goal setting", "personal growth",
            "injury prevention", "recovery", "rehabilitation", "physical therapy", "sports injuries",
            "pain management", "recovery techniques", "foam rolling", "stretching exercises",
            "injury management", "injuries", "apps", "health tracking", "wearable technology", "equipment",
            "home workouts", "gym routines", "outdoor activities", "sports", "wellness tips", "water", "adult", "adults"
            "child", "children", "infant", "sleep", "habit", "habits", "routine", "loose", "weight", "fruits", "vegetables",
            "chicken", "veg", "vegetarian", "non-veg", "non-vegetarian", "plant", "plant-based", "plant based", "fat", "resources",
            "help", "cutting", "bulking", "link", "links", "website", "online", "websites", "peace", "mind", "equipments", "equipment",
            "watch", "tracker", "watch", "band", "height", "injured", "quick", "remedy", "solution", "solutions", "pain"
        ]
        
        greetings=["hello", "hi", "how are you"]

        # check if the input message contains any relevant keywords
        if any(keyword in input_message.lower() for keyword in relevant_keywords):
            response = graph.invoke({
                "messages": [HumanMessage(content=input_message)]
            })
            return response['messages'][1].content
        
        # handle greeting messages
        elif any(keyword in input_message.lower() for keyword in greetings):
            return "Hi there, I am FIT bot, your personal wellbeing coach."
        
        # default response for irrelevant topics
        else:
            return "I'm here to assist with fitness, nutrition, mental health, and related topics. Please ask questions related to these areas."
    except Exception as e:
        return f"An error occurred while processing your request: {e}"


# setup Gradio interface
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

# def update_message(request: gr.Request):
#     return f"Welcome, {request.username}"


# launch the Gradio application

# always use the below given username and password for logging into the web application
demo.launch(auth=("admin", "pass1234"),auth_message="You have succesfully logged in.")
