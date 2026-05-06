import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone  # 1. Use gRPC for speed

load_dotenv()


def get_deterministic_id(content: str, index: int) -> str:
    """Creates a unique, reproducible ID for a chunk."""
    hash_obj = hashlib.md5(content.encode())
    return f"{hash_obj.hexdigest()}-{index}"


def run_indexing():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        output_dimensionality=1536
    )

    loader = TextLoader("vector-db-blog.txt", encoding="utf-8")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=800, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    # 1. Create a separate list of unique IDs
    # This is the step that ensures Pinecone knows these are 24 different records.
    ids = [get_deterministic_id(doc.page_content, i)
           for i, doc in enumerate(split_docs)]
    for idx, unique_id in enumerate(ids):
        print(f"Chunk {idx}: {unique_id}")

    index_name = os.getenv("PINECONE_INDEX_NAME")

    # 2. Use LangChain's from_documents with the 'ids' parameter
    print(f"{ids[:5]}...")  # Print first 5 IDs for verification

    vector_store = PineconeVectorStore(embedding=embeddings,
                                       index_name=index_name,
                                       ids=ids
                                       )

    document_ids = vector_store.add_documents(documents=split_docs)
    
    print(document_ids[:3])

    # vectorstore = PineconeVectorStore.from_documents(
    #     documents=split_docs,
    #     ids=ids,
    #     embedding=embeddings,
    #     index_name=index_name,
    #     namespace="vector-db-blog"
    # )

    print("Indexing complete.")


if __name__ == "__main__":
    run_indexing()
