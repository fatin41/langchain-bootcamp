# LangChain & LangGraph Bootcamp

A comprehensive collection of practical projects and experiments designed to master the LangChain ecosystem. This repository covers everything from foundational Retrieval-Augmented Generation (RAG) to advanced, stateful agentic workflows using LangGraph and Google's Gemini models.

## 🚀 Overview

This project serves as a hands-on sandbox for building intelligent applications. It demonstrates how to orchestrate LLMs, vector databases, and external tools to create systems that can reason, reflect, and adapt to complex queries.

## 🛠️ Tech Stack

- **Frameworks:** [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
- **LLMs:** Google Gemini (`gemini-2.0-flash`)
- **Embeddings:** Google Gemini Embeddings (`models/gemini-embedding-2`)
- **Vector Databases:** [Pinecone](https://www.pinecone.io/) & [ChromaDB](https://www.trychroma.com/)
- **Data Acquisition:** [Tavily AI](https://tavily.com/)
- **Package Manager:** [uv](https://github.com/astral-sh/uv)

---

## 📂 Project Structure & Detailed Topics

The repository is organized into modules that progress from basic chains to complex, stateful agents.

### 1. `rag/` (Core RAG Fundamentals)
**Intention:** Build a foundational understanding of the Retrieval-Augmented Generation lifecycle.
- **Topics Covered:**
    - **Document Loading:** Using `WebBaseLoader` to pull content from the web.
    - **Text Splitting:** Implementing `CharacterTextSplitter` to manage context window limits.
    - **Vector Stores:** Initializing and querying local vector storage.
    - **LCEL (LangChain Expression Language):** Composing chains using the `|` operator for cleaner, more maintainable code.
    - **Manual vs. Automated RAG:** Comparing step-by-step retrieval/generation vs. using pre-built LangChain chains.

### 2. `langchain-helper/` (Production RAG & Ingestion)
**Intention:** Transition from local scripts to a scalable system capable of handling large documentation sets.
- **Topics Covered:**
    - **Scalable Ingestion:** Building an asynchronous pipeline to map and index the entire LangChain documentation site.
    - **Pinecone Integration:** Using `PineconeVectorStore` for persistent, cloud-based vector search.
    - **Advanced Retrieval:** Implementing custom logic to retrieve relevant context from specific namespaces.
    - **API Design:** Structuring a backend (`core.py`) that serves as a query engine for front-end applications.

### 3. `langgraph-course/` (Stateful Agentic Workflows)
**Intention:** Master the use of `LangGraph` to create agents that can loop, reason, and use tools.

#### 🧠 Detailed Breakdown:
- **Agentic RAG (`agentic-rag/`):**
    - **Concept:** Implements "Corrective RAG" (CRAG).
    - **Topics:** `StateGraph` construction, Conditional Edges (routing), Binary Document Grading, and automated fallback to **Tavily Web Search** when local knowledge is insufficient.
- **Reflection Agent (`reflection-agent/`):**
    - **Concept:** A two-node graph (Generator & Reflector).
    - **Topics:** Loop management in graphs, message state handling (`add_messages`), and iterative prompt engineering to force an LLM to critique its own work.
- **Reflexion Agent (`reflexion-agent/`):**
    - **Concept:** Advanced self-correction using structured feedback.
    - **Topics:** Tool-calling with Pydantic schemas, `JsonOutputToolsParser`, and generating dynamic search queries based on identified gaps in initial answers.

### 4. Root Level (Foundation Scripts)
**Intention:** Focused experiments on specific LangChain primitives.
- **Topics Covered:**
    - **Tool Calling:** Binding Python functions to LLMs (`tool-calling.py`).
    - **ReAct Logic:** Deep dive into the "Reason + Act" prompting loop in `raw-react-prompt.py`.
    - **Search Integration:** Standalone use of the Tavily API for real-time web data.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (Recommended) or `pip`

### Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Installation
```bash
# Install dependencies
uv sync
```

## 📖 How to Run

- **Basic RAG:** `python rag/main.py`
- **Ingest Docs:** `python langchain-helper/ingestion.py`
- **Agentic RAG:** `python langgraph-course/agentic-rag/main.py`
- **Reflection Agent:** `python langgraph-course/reflection-agent/main.py`
- **Reflexion Agent:** `python langgraph-course/reflexion-agent/main.py`
