from langgraph.graph import StateGraph, START , END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from typing import Literal
from langchain_mistralai import ChatMistralAI

key = "8zZVzNp9On4lCVnmfyBCQt33qjRCLezH"

model = ChatMistralAI(
    model="mistral-large-2512",
    mistral_api_key= key,  # Direct key
    temperature=0.7
)

class SubGraphState(TypedDict):
    english_text : str 
    chinese_text : str 


def translate_text(State : SubGraphState):
    prompt = f"""Translate the given text into chinese.
    Guadrails : 
    1. Do not change the original given text.
    2.Do not add something into the original text.
    3.Only do the translation.
    "Given text is {State["english_text"]} """
    result = model.invoke(prompt)
    return {"chinese_text": result.content}

graph1 = StateGraph(SubGraphState)

graph1.add_node("translate_text", translate_text)

graph1.add_edge(START, "translate_text")
graph1.add_edge("translate_text", END)

sub_graph = graph1.compile()

class ParentGraph(TypedDict):
    Given_topic : str 
    english_text : str 
    chinese_text : str 

def generate_text(State : ParentGraph):
    prompt = f"Describe the topic {State['Given_topic']} in english language."
    result = model.invoke(prompt)
    return {"english_text" : result.content}

def translate_text(State : ParentGraph):
    result = sub_graph.invoke({"english_text" : State["english_text"]})
    return {"chinese_text": result["chinese_text"]}

graph = StateGraph(ParentGraph)

graph.add_node("generate_text", generate_text)
graph.add_node("translate_text",translate_text)

graph.add_edge(START, "generate_text")
graph.add_edge("generate_text","translate_text")
graph.add_edge("translate_text",END)

graph = graph.compile()

result = graph.invoke({"Given_topic": "Agentic Ai"})

print(result)