# RAG_with_langchain

A Python project for Retrieval-Augmented Generation (RAG) using LangChain, HuggingFace embeddings, Google Gemini, Tavily web search, and local DeepSeek via Ollama.  
It loads PDFs, creates a vector database, and answers cybersecurity questions using both local and web sources.

## Features

- **PDF Loader:** Loads and splits PDF documents into text chunks.
- **Vector Database:** Embeds chunks and stores them in a persistent Chroma vectorstore.
- **Hybrid Retrieval:** Retrieves relevant chunks from local PDFs and web search (Tavily).
- **LLM Integration:** Uses Google Gemini or DeepSeek (via Ollama) to answer questions based on retrieved context.
- **Extensible:** Easily switch embedding models and LLMs.

## Requirements

- Python 3.8+
- [LangChain](https://python.langchain.com/)
- [Chroma](https://www.trychroma.com/)
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
- [HuggingFace Embeddings](https://python.langchain.com/docs/integrations/text_embedding/huggingface/)
- [Google Generative AI](https://github.com/google/generative-ai-python)
- [Tavily Search](https://docs.tavily.com/)
- [Ollama](https://ollama.com/) (for local DeepSeek)
- [requests](https://docs.python-requests.org/)

Install dependencies:
```bash
pip install langchain chromadb pymupdf langchain-google-genai langchain-community langchain-tavily requests
```

## Usage

### 1. Prepare PDFs

Place your PDF files in the `books/` directory (or specify another directory).

### 2. Initialize RAG

```python
from main import RAG

rag = RAG(
    api_key="YOUR_GOOGLE_API_KEY",
    search_key="YOUR_TAVILY_API_KEY",
    pdf_dir="books",
    persist_dir="./chroma_db"
)
```

### 3. Load and Embed PDFs

```python
rag.load_and_embed_pdfs()
```

This will process all PDFs, split them into chunks, embed, and store them in Chroma.

### 4. Ask Questions (Gemini)

```python
answer = rag.ask("What is ransomware?")
print(answer)
```

### 5. Ask Questions (DeepSeek via Ollama)

Make sure Ollama is running and DeepSeek model is available locally.

```python
answer = rag.ask_deepseek_local("What is ransomware?")
print(answer)
```

## Class Overview

### `RAG`

- **__init__**: Sets up API keys, embedding model, LLM, and directories.
- **load_and_embed_pdfs**: Loads PDFs, splits text, embeds, and persists vectorstore.
- **load_vectorstore**: Loads persisted vectorstore from disk.
- **ask**: Retrieves relevant chunks from PDFs and web, combines context, and queries Gemini.
- **ask_deepseek_local**: Same as `ask`, but queries DeepSeek via Ollama API.

## Environment Variables

- `GOOGLE_API_KEY`: Your Google Generative AI API key.
- `TAVILY_API_KEY`: Your Tavily API key.

## Notes

- Ensure your PDF directory exists and contains valid PDF files.
- The vectorstore is persisted in `./chroma_db` by default.
- Tavily and Gemini require valid API keys.
- Ollama must be running for DeepSeek queries.



