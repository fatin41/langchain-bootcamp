import os
import uuid
import datetime
from typing import List, Dict, Any
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# LangChain & Pinecone Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-2"


class PineconePopulation:
    def __init__(self):
        self.pc: Pinecone = Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = os.environ.get("PINECONE_INDEX_NAME")
        self.index = self.pc.Index(name=self.index_name)
        self.batch_size: int = 100

    def populate_to_index(self, data: List[Dict[str, Any]], namespace: str = "") -> None:
        count = 0
        for i in range(0, len(data), self.batch_size):
            batch = data[i: i + self.batch_size]
            try:
                # Upserting raw vectors to the specified namespace
                self.index.upsert(vectors=batch, namespace=namespace)
                print(f"[{datetime.datetime.now(ZoneInfo('Asia/Kolkata'))}] "
                      f"Completed indexing for batch: {count + 1} with batch count {len(batch)}")
                count += 1
            except Exception as e:
                print(f"Error indexing batch starting at index {i}: {e}")
                # Optional: print(f"Sample metadata: {[item['metadata'] for item in batch[:2]]}")
                continue


if __name__ == "__main__":
    # 1. Initialize Embeddings
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=1536
    )

    # 2. Load and Split Documents
    print("Loading and splitting documents...")
    loader = TextLoader("vector-db-blog.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 3. Generate Embeddings & Prepare Data for Pinecone
    print(f"Processing {len(docs)} chunks...")

    formatted_data = []

    # We embed in smaller chunks to avoid API timeouts/errors
    api_batch_size = 20
    for i in range(0, len(docs), api_batch_size):
        batch_docs = docs[i: i + api_batch_size]
        batch_texts = [doc.page_content for doc in batch_docs]

        try:
            # Generate embeddings for this specific batch
            # Workaround: Using embed_query in a loop because embed_documents 
            # might only return one result in some versions/models
            batch_vectors = [embeddings_model.embed_query(text) for text in batch_texts]

            # Map them together immediately
            for j, doc in enumerate(batch_docs):
                payload = {
                    "id": str(uuid.uuid4()),
                    "values": batch_vectors[j],
                    "metadata": {
                        "text": doc.page_content,
                        "source": "vector-db-blog.txt",
                        **doc.metadata
                    }
                }
                formatted_data.append(payload)

            print(f"Embedded chunks {i} to {min(i + len(batch_docs), len(docs))}")

        except Exception as e:
            print(f"Error during embedding generation: {e}")
            continue

    # 4. Use your class to populate the index
    if formatted_data:
        populator = PineconePopulation()
        populator.populate_to_index(formatted_data, namespace="vector-db-blog")
    else:
        print("No data was formatted. Check your Embedding API connection.")

# import os
# import uuid
# from dotenv import load_dotenv
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone

# # Load environment variables from .env
# load_dotenv()

# EMBEDDING_MODEL = "models/gemini-embedding-2"


# def ingest_data():
#     # 1. Setup Keys and Config
#     google_key = os.getenv("GOOGLE_API_KEY")
#     pinecone_key = os.getenv("PINECONE_API_KEY")
#     index_name = os.getenv("PINECONE_INDEX_NAME")

#     if not all([google_key, pinecone_key, index_name]):
#         print("Error: Missing environment variables. Check your .env file.")
#         return

#     # 2. Load and Split Documents
#     print("--- Loading text data ---")
#     try:
#         loader = TextLoader("vector-db-blog.txt", encoding="utf-8")
#         documents = loader.load()
#     except Exception as e:
#         print(f"Error loading file: {e}")
#         return

#     for doc in documents:
#         doc.metadata["source"] = "vector-db-blog.txt"

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100
#     )
#     texts = text_splitter.split_documents(documents)
#     print(f"Created {len(texts)} chunks.")

#     # 3. Initialize Embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model=EMBEDDING_MODEL,
#         output_dimensionality=1536
#     )

#     # 4. Pinecone Index Validation
#     pc = Pinecone(api_key=pinecone_key)
#     index_description = pc.describe_index(index_name)

#     # Gemini-embedding-2 defaults to 1536.
#     if index_description.dimension != 1536:
#         print(
#             f"CRITICAL: Index dimension mismatch! Index is {index_description.dimension}, but model is 1536.")
#         return

#     # 5. Initialize Vector Store
#     vectorstore = PineconeVectorStore(
#         index_name=index_name,
#         embedding=embeddings,
#         namespace="vector-db-blog"
#     )

#     # 6. Batched Ingestion (The Fix)
#     print(f"Starting ingestion into namespace 'vector-db-blog'...")

#     batch_size = 50  # Smaller batches are more reliable for high-dim vectors
#     unique_ids = [str(uuid.uuid4()) for _ in range(len(texts))]

#     for i in range(0, len(texts), batch_size):
#         batch_end = i + batch_size
#         batch_texts = texts[i:batch_end]
#         batch_ids = unique_ids[i:batch_end]

#         vectorstore.add_documents(documents=batch_texts, ids=batch_ids)
#         print(
#             f"Successfully uploaded chunks {i} to {min(batch_end, len(texts))}")

#     print("--- Ingestion Complete ---")

#     # 7. Final Stats Check
#     index = pc.Index(index_name)
#     stats = index.describe_index_stats()
#     print(f"Current Index Stats: {stats}")


# if __name__ == "__main__":
#     ingest_data()
 