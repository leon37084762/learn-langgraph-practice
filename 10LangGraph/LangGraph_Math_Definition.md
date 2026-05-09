# LangGraph 的数学定义

## 一、传统图论的局限

传统有向图的数学定义：

```
G = (V, E)

V = {v₁, v₂, ..., vₙ}      // 节点集合
E ⊆ V × V                   // 有向边集合
```

这个定义只能回答"哪些节点连着哪些节点"，完全无法描述以下关键问题：

- 执行到 `v₂` 时，全局状态 `S` 变成了什么？
- 从 `v₂` 去 `v₃` 还是去 `v₄`，由谁决定、依据什么条件决定？
- 状态在转移过程中如何被修改？

传统图论研究的是**拓扑结构**，而 LangGraph 研究的是**动态执行**。

---

## 二、LangGraph 的完整数学定义

LangGraph 本质上是一个**带状态的计算图（Stateful Computational Graph）**，应定义为**七元组**：

```
ℳ = (V, E, S, δ, γ, σ₀, F)
```

| 符号 | 名称 | 数学定义 | 对应代码 |
|---|---|---|---|
| **V** | 节点集合 | `V = {v₁, v₂, ..., vₙ}` | `graph.add_node("init", init_node)` |
| **E ⊆ V × V** | 普通边集合 | 有向边 `(vᵢ, vⱼ)` | `graph.add_edge("init", "multiply")` |
| **S** | 全局状态空间 | `S = D₁ × D₂ × ... × Dₖ` | `class State(TypedDict): ...` |
| **δ: V × S → S** | 节点状态转移函数 | 节点 `v` 接收状态 `s`，输出新状态 `s'` | `def multiply_node(state): return {...}` |
| **γ: V × S → P(V)** | 条件路由函数 | 节点 `v` + 当前状态 `s` → 决定下一步可以去哪些节点 | `def should_continue(state): return "end"` |
| **σ₀ ∈ S** | 初始状态 | 图的执行起点 | `app.invoke({"value": 5})` |
| **F ⊆ V** | 终止节点集合 | 到达这些节点时图执行结束 | `END` |

---

## 三、实例映射：13LangGraph_no_llm.py

以 `13LangGraph_no_llm.py` 为例，将代码映射到数学符号：

### 3.1 节点集合 V

```python
V = {
    __start__,      // LangGraph 内置起点
    init,
    multiply,
    add,
    final,
    __end__         // LangGraph 内置终点
}
```

### 3.2 普通边集合 E

```python
E = {
    (__start__, init),
    (init, multiply),
    (multiply, add),
    (final, __end__)
}
```

### 3.3 状态空间 S

```python
S = ℤ × ℤ × List(String)   # (value: int, step: int, log: list[str])
```

对应代码：

```python
class State(TypedDict):
    value: int
    step: int
    log: list[str]
```

### 3.4 节点状态转移函数 δ

```
δ(init, (v, _, _))       = (v, 0, ["[init] 初始值: v"])
δ(multiply, (v, k, l))   = (2v, k+1, l + ["[multiply] v × 2 = 2v"])
δ(add, (v, k, l))        = (v+10, k+1, l + ["[add] v + 10 = v+10"])
δ(final, (v, k, l))      = (v, k, l + ["[final] 最终值: v"])
```

对应代码：

```python
def multiply_node(state: State) -> State:
    new_value = state["value"] * 2
    return {
        "value": new_value,
        "step": state["step"] + 1,
        "log": state["log"] + [f"[multiply] ..."]
    }
```

### 3.5 条件路由函数 γ

```
γ(add, (v, _, _)) = 
    {final}       if v ≥ 100
    {multiply}    if v < 100
```

对应代码：

```python
def should_continue(state: State) -> Literal["continue", "end"]:
    if state["value"] >= 100:
        return "end"
    return "continue"
```

### 3.6 初始状态 σ₀

```python
σ₀ = (5, 0, [])   # value=5, step=0, log=[]
```

对应代码：

```python
result = app.invoke({"value": 5})
```

### 3.7 终止节点集合 F

```python
F = {__end__}
```

对应代码：

```python
from langgraph.graph import END
```

---

## 四、关键扩展：为什么 γ 是 V × S → P(V)

这是 LangGraph 和传统**有限状态自动机（FSA）**的根本区别。

| 自动机类型 | 转移函数 | 是否依赖状态 |
|---|---|---|
| **有限状态自动机（FSA）** | `γ: V → P(V)` | ❌ 只看当前节点，不看状态 |
| **Mealy / Moore 机** | `γ: V × Σ → P(V)` | ⚠️ 依赖输入符号，不依赖内部状态 |
| **LangGraph** | `γ: V × S → P(V)` | ✅ **同时依赖节点和全局状态** |

LangGraph 的条件边 `should_continue(state)` 就是一个**状态依赖的路由函数**：

```python
def should_continue(state) -> Literal["continue", "end"]:
    if state["value"] >= 100:   # ← 读取全局状态 S
        return "end"            # → 路由到 {final}
    return "continue"           # → 路由到 {multiply}
```

**核心差异**：FSA 的转移是"节点到节点"的静态映射，LangGraph 的转移是"节点+状态 → 节点"的动态计算。

---

## 五、和传统控制流图（CFG）的区别

编译原理中的控制流图（Control Flow Graph）：

```
CFG = (V, E, entry, exit)
```

### 5.1 边的确定性不同

**CFG 的边是静态的、编译期确定的**：

`if-else` 在 CFG 中会被展开成两条不同的边，运行时只有一条被激活，但**两条边都"存在"在图中**。

```
    [判断节点]
      /    \
   true   false
    /        \
 [分支A]   [分支B]
```

**LangGraph 的边是动态的、运行时确定的**：

```python
graph.add_conditional_edges(
    "add",
    should_continue,        # ← 运行时调用，动态决定
    {"continue": "multiply", "end": "final"}
)
```

同一个节点 `add`，在运行时**可能产生不同的出边**。这在传统图论中没有对应概念，属于**超图（Hypergraph）**或**非确定性转移系统**的范畴。

### 5.2 状态管理不同

**CFG**：没有全局状态概念，变量作用域由编译器管理。

**LangGraph**：显式定义全局状态 `S`，所有节点共享，自动传递。

---

## 六、LangGraph 的动态执行语义

当调用 `app.invoke(σ₀)` 时，LangGraph 的执行引擎执行以下计算：

```
执行轨迹: σ₀ → v₁ → σ₁ → v₂ → σ₂ → ... → vₙ → σₙ

其中:
  v₁ = entry                    // 从入口开始
  σ₁ = δ(v₁, σ₀)               // 节点 v₁ 修改状态
  v₂ ∈ γ(v₁, σ₁)               // 根据新状态决定下一步
  σ₂ = δ(v₂, σ₁)
  ...
  vₙ ∈ F                       // 到达终止节点，执行结束
```

这是一个**状态驱动的非确定性转移系统**，每一步的走向都由当前状态和路由函数共同决定。

---

## 七、总结

> **LangGraph 的图不是拓扑结构，而是一个动态执行的计算模型。**

传统图论 `(V, E)` 研究的是"长什么样"，LangGraph `(V, E, S, δ, γ)` 研究的是"怎么动起来"。

| 维度 | 传统图论 | LangGraph |
|---|---|---|
| 核心问题 | 节点如何连接 | 节点如何执行、状态如何变化、走向如何决定 |
| 边 | 静态拓扑关系 | 动态条件转移 |
| 状态 | 无 | 全局共享、自动传递 |
| 数学模型 | 有向图 `G=(V,E)` | 扩展状态机 `ℳ=(V,E,S,δ,γ,σ₀,F)` |

引入 `δ`（状态转移函数）和 `γ`（条件路由函数），才能把静态的 `(V, E)` 变成可执行的计算模型。这是 LangGraph 从"图"跃升为"图计算引擎"的数学本质。
