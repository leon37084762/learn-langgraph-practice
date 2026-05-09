from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END

# ========== 1. 定义状态 ==========
class State(TypedDict):
    """全局状态：所有节点共享这个数据容器"""
    value: int      # 当前数值
    step: int       # 执行步数
    log: list[str]  # 操作日志


# ========== 2. 定义节点函数 ==========
def init_node(state: State) -> State:
    """初始化节点：设置初始值和日志"""
    return {
        "value": state.get("value", 5),
        "step": 0,
        "log": [f"[init] 初始值: {state.get('value', 5)}"]
    }

def multiply_node(state: State) -> State:
    """翻倍节点：将 value 乘以 2"""
    new_value = state["value"] * 2
    return {
        "value": new_value,
        "step": state["step"] + 1,
        "log": state["log"] + [f"[multiply] {state['value']} × 2 = {new_value} (step {state['step'] + 1})"]
    }

def add_node(state: State) -> State:
    """加值节点：将 value 加上 10"""
    new_value = state["value"] + 10
    return {
        "value": new_value,
        "step": state["step"] + 1,
        "log": state["log"] + [f"[add] {state['value']} + 10 = {new_value} (step {state['step'] + 1})"]
    }

def final_node(state: State) -> State:
    """结束节点：输出最终结果"""
    return {
        "log": state["log"] + [f"[final] 最终值: {state['value']}, 总步数: {state['step']}"]
    }


# ========== 3. 定义条件路由 ==========
def should_continue(state: State) -> Literal["continue", "end"]:
    """条件判断：value >= 100 则结束，否则继续循环"""
    if state["value"] >= 100:
        return "end"
    return "continue"


# ========== 4. 构建图 ==========
graph = StateGraph(State)

# 注册节点
graph.add_node("init", init_node)
graph.add_node("multiply", multiply_node)
graph.add_node("add", add_node)
graph.add_node("final", final_node)

# 注册边
graph.add_edge(START, "init")           # START → init
graph.add_edge("init", "multiply")      # init → multiply
graph.add_edge("multiply", "add")       # multiply → add

# 条件边：从 add 节点出发，根据 should_continue 决定走向
graph.add_conditional_edges(
    "add",
    should_continue,
    {
        "continue": "multiply",  # 继续循环：回到 multiply
        "end": "final"           # 结束：走向 final
    }
)

graph.add_edge("final", END)            # final → END


# ========== 5. 编译并执行 ==========
if __name__ == "__main__":
    # 编译图（生成可执行的工作流）
    app = graph.compile()

    print("=" * 60)
    print("LangGraph 无 LLM 示例：数值处理流水线")
    print("规则：初始值 → 翻倍 → 加10 → 判断(≥100?) → 循环或结束")
    print("=" * 60)

    # 执行图，传入初始状态
    result = app.invoke({"value": 5})

    # 打印执行日志
    print("\n📋 执行日志：")
    for line in result["log"]:
        print(f"  {line}")

    print(f"\n✅ 最终结果: value={result['value']}, step={result['step']}")

    # 可视化图结构（Mermaid 语法）
    print("\n📊 图结构可视化（Mermaid 语法）：")
    try:
        print(app.get_graph().draw_mermaid())
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")
        # 兜底输出：直接打印节点和边
        graph_obj = app.get_graph()
        print(f"\n  节点列表: {list(graph_obj.nodes.keys())}")
        print(f"  边列表: {[(s, t) for s, t in graph_obj.edges]}")
        print(f"  条件边: {[(s, name) for s, name, _ in graph_obj.branches]}")
