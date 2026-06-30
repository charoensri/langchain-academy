from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. Define shared state
class State(TypedDict):
    query: str
    weather_data: str
    analysis: str

# 2. Define nodes (steps)
def fetch_weather(state: State):
    # pretend API call
    data = f"Weather for {state['query']}: 25C, sunny"
    return {**state, "weather_data": data}

def analyze_weather(state: State):
    analysis = f"Analysis: {state['weather_data']} is pleasant"
    return {**state, "analysis": analysis}

def generate_response(state: State):
    result = f"Final: {state['analysis']}"
    return state | {"result": result}

# 3. Build graph (THIS is the core difference)
builder = StateGraph(State)
builder.add_node("fetch", fetch_weather)
builder.add_node("analyze", analyze_weather)
builder.add_node("respond", generate_response)

# Define flow explicitly
builder.set_entry_point("fetch")
builder.add_edge("fetch", "analyze")
builder.add_edge("analyze", "respond")
builder.add_edge("respond", END)
graph = builder.compile()

# 4. Run
#output = graph.invoke({"query": "Sydney"})
#print(output["result"])