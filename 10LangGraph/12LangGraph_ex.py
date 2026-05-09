from typing import Literal, TypedDict
from langgraph.graph import StateGraph,START,END
import random

class State(TypedDict):
    graph_state:str
def node_1(state):
    return {"graph_state":state['graph_state'] + " I am"}

def decide_mood(state)->Literal["happy","sad","neutral"]:
    return random.choice(["happy","sad","neutral"])

graph = StateGraph(State)
graph.add_condition_edge("decide_mood",decide_mood)
graph.add_node("node_1",node_1)
graph.add_edge(START,"node_1")
