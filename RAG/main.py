import os
import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
import google.generativeai as genai
import asyncio
import requests

class RAG:
    def __init__(self, api_key,search_key,pdf_dir = "books",persist_dir="./chroma_db", embedding_model="all-MiniLM-L6-v2", llm_model="gemini-2.0-flash"):
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["TAVILY_API_KEY"] = search_key
        genai.configure(api_key=api_key)
        self.pdf_dir = pdf_dir
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        self.model = genai.GenerativeModel(llm_model)
        self.vectorstore = None

    def load_and_embed_pdfs(self):
        all_texts = []
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith(".pdf"):
                path = os.path.join(self.pdf_dir, filename)
                print(f"[INFO] Loading {filename}")
                try:
                    doc = fitz.open(path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    chunks = self.splitter.split_text(text)
                    all_texts.extend(chunks)
                except Exception as e:
                    print(f"[WARN] Failed to load {filename}: {e}")

        print(f"[INFO] Total chunks loaded: {len(all_texts)}")
        self.vectorstore = Chroma.from_texts(all_texts, self.embeddings, persist_directory=self.persist_dir)
        self.vectorstore.persist()
        print("[INFO] Vectorstore created and persisted.")

    def load_vectorstore(self):
        if os.path.exists(self.persist_dir):
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            print("[INFO] Vectorstore loaded from disk.")
        else:
            print("[WARN] Vectorstore directory not found. Run load_and_embed_pdfs() first.")

    def ask(self, query, top_k=2, distance_threshold=0.25):
        self.load_vectorstore()

        use_rag = False
        pdf_docs = []

        if self.vectorstore:
            # This returns [(Document, distance), ...]
            pdf_docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)

            if pdf_docs_with_scores:
                min_distance = min(score for _, score in pdf_docs_with_scores)
                use_rag = min_distance <= distance_threshold  # lower distance = more relevant

                # Keep only the documents
                pdf_docs = [doc for doc, _ in pdf_docs_with_scores]

        if use_rag:
            # Retrieve from Tavily too
            tavily_retriever = TavilySearch()
            web_docs = tavily_retriever.invoke({"query": query, "num_results": top_k})

            # Merge context
            combined_texts = [getattr(d, "page_content", "") for d in pdf_docs] + web_docs
            context = "\n\n".join([text for text in combined_texts if text.strip()])

            system_prompt = "You are a cybersecurity assistant. Analyze the combined content below and answer the question."

            final_prompt = f"""
            {system_prompt}

            Content:
            {context}

            Question: {query}
            Answer:
            """
        else:
            # Skip RAG and just answer normally
            final_prompt = query
            print("[INFO] Skipping RAG — low relevance in KB.")

        try:
            response = self.model.generate_content(final_prompt)
            return response.text
        except Exception as e:
            return f"Error talking to Gemini: {e}"
