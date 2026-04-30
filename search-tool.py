from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# Import the pre-built Tavily tool
from langchain_tavily import TavilySearch

load_dotenv()

def main():
    print("Hello from real search tool!")

    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # 2. Initialize the pre-built tool
    # We no longer need the @tool decorator, LangChain gives us the tool ready to use
    search = TavilySearch(max_results=2)
    tools = [search]

    # 3. Create the Agent
    agent = create_agent(
        llm,
        tools,
        system_prompt="You are a helpful assistant that uses the search tool to answer questions using the internet. Summarize the results."
    )

    # 4. Invoke the Agent
    inputs = {"messages": [("user", "What is the latest news regarding SpaceX today?")]}

    print("Agent is thinking and searching the web...")
    result = agent.invoke(inputs)

    # 5. Print the final message from the conversation
    final_message = result["messages"][-1]

    content = final_message.content
    # Clean up output if Gemini returns a list of blocks
    if isinstance(content, list):
        text_parts = [part['text']
                      for part in content if isinstance(part, dict) and 'text' in part]
        content = " ".join(text_parts) if text_parts else str(content)

    print("\nFinal Answer:\n", content)


if __name__ == "__main__":
    main()
