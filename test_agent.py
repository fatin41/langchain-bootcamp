from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def dummy(): "dummy"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
agent = create_agent(llm, tools=[dummy])
res = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
print(res)
