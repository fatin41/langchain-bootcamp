from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral instagram influencer grading instagram captions. Generate critique and recommendations for the user's latest caption attempts. Always provide detailed recommendations including requests for length, virality, style etc."
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "Please critique the latest assistant response above.")
    ]
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a instagram techie influencer assitant tasked with writing excellent captions for posts. Generat the best instagram captions for the user's request. If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
