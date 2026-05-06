import asyncio
import os
import ssl
from typing import List, Dict, Any
import certifi

# LangChain & Pinecone Imports
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap, tavily_extract
from logger import (Colors, log_info, log_error,
                    log_warning, log_success, log_header)

from dotenv import load_dotenv
load_dotenv()


# configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10,
    output_dimensionality=1536
)

vectorstore = Pineconevectorstore = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
    embedding=embeddings,
    namespace="langchain-docs"
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavlily_crawl = TavilyCrawl(extract=tavily_extract, map=tavily_map)
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=500)

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract content from a batch of URLs using TavilyExtract"""
    log_info(
        f"Extracting batch {batch_num} with {len(urls)} URLs...", Colors.BLUE)
    try:
        extraction_res = tavily_extract.invoke({"urls": urls})
        log_success(
            f"Batch {batch_num} extraction complete! Extracted {len(extraction_res.get('results', []))} pages.")
        return extraction_res.get('results', [])
    except Exception as e:
        log_error(f"Error extracting batch {batch_num}: {str(e)}")
        return []


def chunk_urls(urls: List[str], batch_size: int = 20) -> List[List[str]]:
    """Split list of URLs into batches"""
    chunks = []
    for i in range(0, len(urls), batch_size):
        chunk = urls[i:i + batch_size]
        chunks.append(chunk)
    return chunks


async def index_documents_async(docs: List[Document], batch_size: int = 100):
    """Index documents into Pinecone in batches"""
    log_header("Pinecone Indexing Phase")
    log_info(
        f"Indexing {len(docs)} chunks into Pinecone in batches of {batch_size}...")

    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
    log_info(f"Created {len(batches)} batches for indexing.")

    # process all batches asynchronously to speed up indexing
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            vectorstore.add_documents(batch)
            log_success(f"Batch {batch_num} indexed successfully!")
        except Exception as e:
            log_error(f"Error indexing batch {batch_num}: {str(e)}")
            return False
        return True

    # process all batches concurrently
    tasks = [add_batch(batch, idx + 1) for idx, batch in enumerate(batches)]
    result = await asyncio.gather(*tasks, return_exceptions=True)

    # count successes
    successful = sum(1 for res in result if res is True)
    if successful == len(batches):
        log_success(
            "All batches indexed successfully! Your RAG is ready for queries.")
    else:
        log_warning(
            f"{successful} out of {len(batches)} batches indexed successfully. Please check logs for errors.")


async def async_extract(url_batches: List[List[str]]):
    """Run asynchronous extraction for all batches"""
    log_header("Document Extraction Phase")
    tasks = [extract_batch(batch, idx + 1)
             for idx, batch in enumerate(url_batches)]
    result = await asyncio.gather(*tasks, return_exceptions=True)

    # filter out any exceptions and flatten the list of results
    extracted_docs = []
    failed_batches = 0
    for res in result:
        if isinstance(res, Exception):
            log_error(f"Extraction error: {str(res)}")
            failed_batches += 1
        else:
            for extracted_page in res:
                extracted_docs.append(Document(
                    page_content=extracted_page.get('raw_content', ''),
                    metadata={"source": extracted_page.get(
                        'url'), "title": extracted_page.get('title')}
                ))

    log_success(
        f"Extraction complete! Successfully extracted {len(extracted_docs)} documents with {failed_batches} failed batches.")

    if failed_batches > 0:
        log_warning(
            f"TavilyExtract had issues with {failed_batches} batches. ")

    return extracted_docs


async def main():
    log_header("Documentation Ingestion Process Started")

    # 1. Map the Site: Get all relevant documentation URLs
    log_info("Mapping the documentation site...",
             Colors.PURPLE)

    site_map = tavily_map.invoke(
        "https://docs.langchain.com/"
    )

    log_success(
        f"Mapping complete! Found {len(site_map.get('results', []))} pages to process.")

    # split url into batches of 20 for extraction
    url_batches = chunk_urls(list(site_map.get('results')), batch_size=20)

    log_info(
        f"Processing {len(url_batches)} batches of URLs for extraction...",
        Colors.BLUE
    )

    # extract documents from urls
    all_docs = await async_extract(url_batches)

    # split documents into chunks for indexing
    log_header("Document Chunking Phase")
    log_info(f"Chunking {len(all_docs)} extracted documents for indexing...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Chunking complete! Created {len(splitted_docs)} chunks from {len(all_docs)} documents.")

    await index_documents_async(splitted_docs, batch_size=500)

    log_header("Ingestion Process Completed!")
    log_success(
        "Your RAG system is now ready for queries. You can test it with a sample query to see how it performs.")
    log_info("Summery", Colors.BOLD)
    log_info(f"Total Pages Mapped: {len(site_map.get('results', []))}")
    log_info(f"Total Documents Extracted: {len(all_docs)}")
    log_info(f"Total Chunks Created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
