from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from google.genai import types
from google import genai
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-2"

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL,
                                          output_dimensionality=1536)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

vectorstore = PineconeVectorStore(index_name=os.environ.get(
    "PINECONE_INDEX_NAME"), embedding=embeddings, namespace="vector-db-blog")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    """You are an expert assistant. Use the provided context to answer the user's question. 
    If the answer isn't explicitly in the context but the context gives clues, use those clues.
    
    Context:
    {context} 
    
    Question: {question}
    
    Detailed Answer:"""
)

def format_docs(docs):
    """Format retrieved documents with clear XML-style tags for the LLM."""
    formatted = []

    for i, doc in enumerate(docs):
        # Using XML-style tags like <doc> helps LLMs distinguish between chunks
        source = doc.metadata.get('source', 'Unknown')

        chunk_text = (
            f"<document index='{i}'>\n"
            f"Source: {source}\n"
            f"Content: {doc.page_content}\n"
            f"</document>"
        )
        formatted.append(chunk_text)

    final_context = "\n\n".join(formatted)

    # Optional: Print for debugging
    print(f"\n--- Retrieved {len(docs)} chunks ---")

    return final_context


def retrieval_change_without_langchain_expression(question: str):
    """ Simple retrieval augmented generation without using LCEL.

    Manually retreive relevant documents, formats, and generate response.

    Limitations
     - Manual step by step execution.
     - No built-in streaming support.
     - No Asynchronous support.
     - Harder to compose with other tools or chains.
     - More verbose and less elegant than using LangChain's agent framework.
    """

    # Step 1: Retrieve relevant documents based on the question
    docs = retriever.invoke(question)

    # Step 2: Format the retrieved documents into a string to be included in the prompt
    context = format_docs(docs)

    # Step 3: Create the final prompt by filling in the context and question
    final_prompt = prompt.format(context=context, question=question)
    print("Final Prompt:\n", final_prompt)

    # step 4: Pass the final prompt to the LLM to get the answer
    response = llm.invoke([HumanMessage(content=final_prompt)])
    return response


def create_retrieval_chain(question: str):
    """ Create a simple retrieval augmented generation chain using LangChain's prompt and LLM.

    This is a more elegant and composable way to create a RAG chain compared to the manual approach.
    """
    # Step 1: Create a retrieval chain that combines the retriever and the prompt
    # retrieval_chain = retriever | prompt | llm

    retrievel_chain = (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs) |
        prompt |
        llm |
        StrOutputParser()
    )

    # Step 2: Invoke the chain with the question to get the answer
    # response = retrieval_chain.invoke(question)

    return retrievel_chain


if __name__ == "__main__":
    question = "What are Embeddings?"

    ########################
    # Invoke Without RAG
    #########################
    # result_raw = llm.invoke([HumanMessage(content=question)])
    # print("\nRAW Answer:\n", result_raw.content)

    # print("\n" + "="*20 + "\n")
    ########################
    # Invoke Without LCEL
    #########################
    # result_rag = retrieval_change_without_langchain_expression(question)
    # print("\nRAG Answer Without LangChain:\n", result_rag.content)

    ########################
    # Invoke with LCEL
    #########################
    chain_with_lcel = create_retrieval_chain(question)
    result_with_lcel = chain_with_lcel.invoke({"question": question})

    print("\nRAG Answer with LCEL:\n", result_with_lcel)
