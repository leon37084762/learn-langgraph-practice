from typing import TypedDict,Annotated
from typing_extensions import Literal

print("== LangGraph Concept ===\n")

# 1.state schema 状态定义
class AgentState(TypedDict):
    """ Agent State define """
    messages:list[dict]
    intent:str
    next_node:str

print("1.State Schema:")

print("""
    class AgentState(TypeDict):
        messages:list[dict]
            intent:str
            next_node:str
""")

# 2.Nodes 节点函数
def classify_intent(state:AgentState) -> AgentState:
    """ 意图分类节点 """
    user_message = state["messages"][-1]["content"]
    if "天气" in user_message:
        state["intent"] = "weather"
        state["next_node"] = "weather_tool"
    else:
        state["intent"] = "general"
        state["next_node"] = "llm"
    return state

# 3.Edges 边定义
print("\n3.Edges:")
print(""" 
    - Normal Edge(普通边)
        graph.add_edge("node_a","node_b")
    - Conditional Edge(条件边)
        graph.add_conditional_edge(
            "classifier",
            route_function,
            {
                "weather": "weather_tool",
                "general": "llm"
            }
        )
""")

# 4. Graph 结构
print("\n4.Graph Structure:")
print("""
                 START
                  ↓
            [Classify Intent]
                  ↓
      ┌─────┴─────┐
      ↓                      ↓
  [Weather]                 [General]
      ↓                      ↓
      └─────┬─────┘
                  ↓
                 END
""")

#5. LangGraph vs LangChain
print("\n5.LangGraph vs LangChain:")
print(""" 

""")