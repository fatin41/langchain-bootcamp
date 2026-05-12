from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated
from chains import generate_chain, reflect_chain

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, path_map={
                              END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

if __name__ == "__main__":
    print("Hello reflection-agent")
    inputs = HumanMessage(
        content="""Make this instagram caption better: This is how a weekend ride ended in mountains""")

    response = graph.invoke(input={"messages": [inputs]})
    print(response)
