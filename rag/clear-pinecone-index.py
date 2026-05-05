import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables (PINECONE_API_KEY, PINECONE_INDEX_NAME, GOOGLE_API_KEY)
load_dotenv()

# We need the embedding model to initialize the VectorStore object
EMBEDDING_MODEL = "models/gemini-embedding-2"


def clear_pinecone_namespace(namespace_name: str):
    """
    Connects to the specified Pinecone index and deletes all vectors 
    within a specific namespace.
    """
    print(f"Initializing connection to delete namespace: {namespace_name}...")

    # 1. Initialize embeddings (must match your ingestion/retrieval dimensionality)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=1536
    )

    # 2. Initialize the VectorStore
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not index_name:
        print("Error: PINECONE_INDEX_NAME not found in environment variables.")
        return

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace_name
    )

    # 3. Perform the deletion
    try:
        print(
            f"Attempting to clear all vectors in namespace: '{namespace_name}'...")
        # delete_all=True tells Pinecone to wipe the specific namespace provided above
        vectorstore.delete(delete_all=True)
        print(f"Successfully cleared namespace: {namespace_name}")
    except Exception as e:
        print(f"An error occurred while clearing the namespace: {e}")


if __name__ == "__main__":
    # Change this if you used a different namespace name
    # __default__
    # vector-db-blog
    TARGET_NAMESPACE = "vector-db-blog"

    confirm = input(
        f"Are you sure you want to delete ALL data in the '{TARGET_NAMESPACE}' namespace? (y/n): ")
    if confirm.lower() == 'y':
        clear_pinecone_namespace(TARGET_NAMESPACE)
    else:
        print("Operation cancelled.")
