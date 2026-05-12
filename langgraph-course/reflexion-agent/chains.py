import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.messages import HumanMessage
from schemas import AnswerQuestion
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher. 
            Current time: {time} 
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.
            """
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide 250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about Reflection agents and Reflexion Agents, what are they how they work. make this into a linkedin Post."
    )

    chain = (first_responder_prompt_template
             | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
             | parser_pydantic
             )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
