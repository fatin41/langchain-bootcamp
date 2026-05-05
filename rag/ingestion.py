from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-2"

if __name__ == "__main__":
    print("loading text data...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "vector-db-blog.txt")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # Ensure source metadata is present
    for doc in documents:
        doc.metadata["source"] = "vector-db-blog.txt"

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Number of chunks created: {len(texts)}")

    # DEBUG: Print a sample of the second or third chunk to verify content
    if len(texts) > 1:
        print(f"Sample content from Chunk 2: {texts[1].page_content[:50]}...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=1536
    )

    index_name = os.environ.get("PINECONE_INDEX_NAME")

    print(
        f"Initializing Pinecone Index: {index_name} with namespace: vector-db-blog")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace="vector-db-blog"
    )

    print("Ingesting data into Pinecone (this may take a moment)...")
    doc_ids = vectorstore.add_documents(texts)
    print(f"Successfully ingested {len(doc_ids)} chunks into Pinecone!")
