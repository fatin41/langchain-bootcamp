from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools


load_dotenv()


SYSTEM_MESSAGE = """
You are a helpful assistant. You have access to tools.
You MUST use the `tavily_search` tool to look up information if you are asked about the weather, current events, or any factual information you are not 100% sure about. 
Do not decline to answer; use the search tool.
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}] + state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)
