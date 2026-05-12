from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """
        param num: a number to triple
        returns: the triple of the input number
    """

    return float(num) * 3


tavily_tool = TavilySearch(max_results=1)


@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a specific location.
    param location: The city or location to get the weather for.
    returns: A weather report.
    """
    return tavily_tool.invoke({"query": f"weather in {location}"})


tools = [tavily_tool, triple, get_weather]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0).bind_tools(tools)
