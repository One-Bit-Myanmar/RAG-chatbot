import os
import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
import google.generativeai as genai
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

    def ask(self, query, top_k=4):
        self.load_vectorstore()
        # 2. Retrieve from local vectorstore
        if self.vectorstore:
            pdf_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            pdf_docs = pdf_retriever.get_relevant_documents(query)
        else:
            pdf_docs = []

        # 3. Retrieve from Tavily (web search)
        tavily_retriever = TavilySearch()
        web_docs = tavily_retriever.invoke({"query": query, "num_results": top_k})

        combined_texts = []

        for d in pdf_docs:
            combined_texts.append(getattr(d, "page_content", ""))

        for d in web_docs:
            combined_texts.append(d)

        context = "\n\n".join([text for text in combined_texts if text.strip()])


        prompt = f"""
        You are a cybersecurity assistant. Analyze the combined content below and answer the question.

        Content:
        {context}

        Question: {query}
        Answer:
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error talking to Gemini: {e}"
        
    def ask_deepseek_local(self,query,top_k = 5,ollama_url="http://localhost:11434"):
        
        self.load_vectorstore()
        # 2. Retrieve from local vectorstore
        if self.vectorstore:
            pdf_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            pdf_docs = pdf_retriever.get_relevant_documents(query)
        else:
            pdf_docs = []

        # 3. Retrieve from Tavily (web search)
        tavily_retriever = TavilySearch()
        web_docs = tavily_retriever.invoke({"query": query, "num_results": top_k})

        # 4. Combine documents
        combined_texts = []

        for d in pdf_docs:
            combined_texts.append(getattr(d, "page_content", ""))

        for d in web_docs:
            combined_texts.append(d)

        context = "\n\n".join([text for text in combined_texts if text.strip()])
        print("----------------------------------------------------")
        print(context)
        prompt = f"""
        You are a cybersecurity assistant. Analyze the combined content below and answer the question.

        Content:
        {context}

        Question: {query}
        Answer:"""
        
        
        # 5. Call Deepseek via Ollama API (/api/chat for chat models)
        url = f"{ollama_url}/api/chat"
        payload = {
            "model": "deepseek-r1",  # must match your local model name
            "messages": [
                {"role": "user", "content": "hello"}
            ]
        }

        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # The exact key depends on Ollama's response format, usually:
            return data.get("completion", "No completion found")
        except Exception as e:
            return f"Error calling Deepseek: {e}"