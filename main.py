import os
import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
import google.generativeai as genai

class RAG:
    def __init__(self, api_key,search_key,pdf_dir = "#",persist_dir="./chroma_db", embedding_model_name="models/embedding-001", llm_model="gemini-2.0-flash"):
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["TAVILY_API_KEY"] = search_key
        genai.configure(api_key=api_key)
        self.pdf_dir = pdf_dir
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model_name
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
        self.vectorstore = Chroma.from_texts(all_texts, self.embedding_model, persist_directory=self.persist_dir)
        self.vectorstore.persist()
        print("[INFO] Vectorstore created and persisted.")

    def load_vectorstore(self):
        if os.path.exists(self.persist_dir):
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embedding_model)
            print("[INFO] Vectorstore loaded from disk.")
        else:
            print("[WARN] Vectorstore directory not found. Run load_and_embed_pdfs() first.")

    def ask(self, query, top_k=4):
        self.load_vectorstore()
        # retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        # docs = retriever.get_relevant_documents(query)
        retriever = TavilySearchResults(k=5)
        docs = self.retriever.run(query)
        print(docs)
        context = "\n\n".join([doc['content'] for doc in docs if 'content' in doc])

        prompt = f"""
        You are a cybersecurity assistant. analyze the content from given contents to answer the question.

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
