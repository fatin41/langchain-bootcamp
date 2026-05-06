# LangChain Bootcamp Projects

A comprehensive collection of LangChain-based applications and experiments, focusing on Retrieval-Augmented Generation (RAG), Agentic workflows, and tool integration using the Google Gemini ecosystem.

## 🚀 Overview

This repository contains several sub-projects and experimental scripts developed to master LangChain. It demonstrates how to build robust RAG systems, perform large-scale documentation ingestion, and create intelligent agents that can use tools.

## 🛠️ Tech Stack

- **Framework:** [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
- **LLMs:** Google Gemini (`gemini-2.0-flash`)
- **Embeddings:** Google Gemini Embeddings (`models/gemini-embedding-2`)
- **Vector Database:** [Pinecone](https://www.pinecone.io/)
- **Data Acquisition:** [Tavily AI](https://tavily.com/) (for web mapping and content extraction)
- **Package Manager:** [uv](https://github.com/astral-sh/uv)

## 📂 Project Structure

The repository is organized into distinct modules:

### 1. `langchain-helper/` (Advanced RAG & Agents)
This is a sophisticated RAG implementation focused on LangChain's own documentation.
- **`ingestion.py`**: An asynchronous pipeline that maps `docs.langchain.com` using Tavily, extracts content, chunks it, and indexes it into a Pinecone namespace (`langchain-docs`).
- **`backend/core.py`**: Implements a RAG agent capable of answering complex questions about LangChain by retrieving relevant context from the vector store.

### 2. `rag/` (Core RAG Implementation)
A focused implementation of a RAG pipeline based on a specific blog post.
- **`ingestion.py`**: Handles loading, chunking, and indexing of `vector-db-blog.txt`.
- **`main.py`**: Showcases two ways to build RAG:
    - Manual retrieval and formatting (Step-by-step).
    - Elegant implementation using **LCEL (LangChain Expression Language)**.

### 3. Root Level (Experimental Scripts)
Various scripts exploring specific LangChain features:
- `tool-calling.py` / `raw-tool-calling.py`: Experiments with LLM tool invocation.
- `raw-react-prompt.py`: Implementation of the ReAct (Reasoning and Acting) pattern.
- `search-tool.py`: Integration of external search capabilities.
- `clear-pinecone-index.py`: Utility to reset the vector database.

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) installed.

### Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
TAVILY_API_KEY=your_tavily_api_key
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd langchain-bootcamp

# Install dependencies using uv
uv sync
```

## 📖 Usage

### Documentation Ingestion (LangChain Docs)
To crawl and index the LangChain documentation:
```bash
python langchain-helper/ingestion.py
```

### Running the RAG Agent
To query the indexed LangChain documentation:
```bash
# Navigate to the helper directory or run from root
python langchain-helper/backend/core.py
```

### Blog-based RAG
To index the blog post and run queries:
```bash
cd rag
python ingestion.py
python main.py
```

## 🧪 Questions for Developers/Contributors
*To make this README even more detailed, please clarify:*
1. Is there a specific Pinecone index configuration (dimension, metric) recommended for `gemini-embedding-2` (e.g., 1536 dimensions, Cosine similarity)?
2. Should any of the root-level scripts be considered "deprecated" in favor of the `langchain-helper` version?
3. Is there a frontend planned, or is this intended strictly as a backend/CLI set of tools?
