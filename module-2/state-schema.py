
# %% [markdown]
# [Pydantic](https://docs.pydantic.dev/latest/api/base_model/) is a data validation and settings management library using Python type annotations. 
# 
# It's particularly well-suited [for defining state schemas in LangGraph](https://docs.langchain.com/oss/python/langgraph/use-graph-api#use-pydantic-models-for-graph-state) due to its validation capabilities.
# 
# Pydantic can perform validation to check whether data conforms to the specified types and constraints at runtime.

# %%

import pprint
import random
from typing import Literal
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, field_validator, ValidationError

class PydanticState(BaseModel):
    name: str
    mood: str # "happy" or "sad" 
    fullSentence: str

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        # Ensure the mood is either "happy" or "sad"
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value


def node_1(state):
    print("---Node 1---")
    return {"name": state.name+ " is ... "}

def node_2(state):
    print("---Node 2---")
    state.mood = "happy"
    return {"fullSentence":  state.name + state.mood}

def node_3(state):
    print("---Node 3---")
    #return {"mood":  "sad"}
    state.mood = "sad"
    return {"fullSentence":  state.name + state.mood}

def decide_mood(state) -> Literal["node_2", "node_3"]:
        
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"


# try:
#     state = PydanticState(name="John Doe", mood="mad")
# except ValidationError as e:
#     print("Validation Error:", e)




# %% [markdown]
# We can use `PydanticState` in our graph seamlessly. 

# %%
# Build graph
builder = StateGraph(PydanticState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

# %%
response = graph.invoke(PydanticState(name="Lance",mood="sad", fullSentence="" ))
print(response)

# %%
# PS C:\Users\chars1\OneDrive - Pegasystems Inc\python\LangChain\langchain-academy\langchain-academy\module-2> python state-schema.py
# <IPython.core.display.Image object>
# ---Node 1---
# ---Node 3---
# {'name': 'Lance is ... ', 'mood': 'sad'}


# PS C:\Users\chars1\OneDrive - Pegasystems Inc\python\LangChain\langchain-academy\langchain-academy\module-2> python state-schema.py
# <IPython.core.display.Image object>
# ---Node 1---
# ---Node 2---
# {'name': 'Lance is ... ', 'mood': 'sad', 'fullSentence': 'Lance is ... happy'}

