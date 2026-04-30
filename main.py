from urllib import response
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()


# Lesson 2: Agent Creation
@tool
def search(query: str) -> str:
    """
    A simple search tool that seaches over internet.
    
    Args:
        query: The search query string
        
    Returns:
        str: A search result
    """
    # In a real implementation, this would call an external search API.
    print(f"searching for: {query}")
    return "Delhi weather is sunny"


def main():

    print("Hello from langchain-bootcamp!")

    #llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    tools = [search]

    agent = create_agent(llm=llm, tools=tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{topic}"),
        ]
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    response = chain.invoke("Explain quantum physics in one sentence.")
    print(response)


if __name__ == "__main__":
    main()



# # Lesson1
# def main():

    
#     print("Hello from langchain-bootcamp!")

#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant."),
#             ("human", "{topic}"),
#         ]
#     )

#     output_parser = StrOutputParser()

#     chain = prompt | llm | output_parser

#     response = chain.invoke("Explain quantum physics in one sentence.")
#     print(response)



