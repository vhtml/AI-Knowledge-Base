from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# 1. 状态：节点之间共享的数据
class State(TypedDict):
    name: str
    greeting: str
    category: str


# 2. 节点：接收状态，返回需要更新的字段
def greet(state: State):
    return {"greeting": f"你好，{state['name']}！欢迎学习 LangGraph。"}

def category(state: State):
    return {"category": "category " + state["name"]}

# 3. 连线：定义执行顺序
builder = StateGraph(State)
builder.add_node("greet", greet)
builder.add_edge(START, "greet")
builder.add_node("category", category)
builder.add_edge("greet", "category")
builder.add_edge("category", END)

# 4. 编译并运行
graph = builder.compile()
result = graph.invoke({"name": "小明"})

print(result)

print(result["greeting"])