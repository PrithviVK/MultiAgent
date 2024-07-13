import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Sequence
from agents import supervisor_chain,nutritionist_node,workout_coach_node,mental_health_coach_node,members


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", action=supervisor_chain)
    workflow.add_node("nutritionist", action=nutritionist_node)
    workflow.add_node("workout_coach", action=workout_coach_node)
    workflow.add_node("mental_health_coach", action=mental_health_coach_node)

    for member in members:
        workflow.add_edge(start_key=member, end_key="supervisor")

    # members = ["nutritionist", "workout_coach", "mental_health_coach"]
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END


    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    workflow.add_edge(START, "supervisor")

    graph= workflow.compile()

    return graph
