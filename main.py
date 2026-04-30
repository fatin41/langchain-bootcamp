from urllib import response
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def main():
    print("Hello from langchain-bootcamp!")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    promptTemplate = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{topic}"),
        ]
    )

    output_parser = StrOutputParser()

    chain = promptTemplate | llm | output_parser

    response = chain.invoke("Explain quantum physics in one sentence.")
    print(response)


    print(type(response)) 

if __name__ == "__main__":
    main()
