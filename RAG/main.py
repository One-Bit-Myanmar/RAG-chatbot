import os
import fitz
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.schema import Document


class RAG:
    def __init__(
        self,
        search_key,
        pdf_dir="books",
        persist_dir="./chroma_db",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="deepseek-r1",
    ):
        self.pdf_dir = pdf_dir
        self.persist_dir = persist_dir
        os.environ["TAVILY_API_KEY"] = search_key
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.vectorstore = None
        self.load_vectorstore()

        # Multi-user memory
        self.user_memory = {}  # user_id -> ConversationBufferMemory
        self.llm = Ollama(model=llm_model)

        # Prompt + LLMChain for StuffDocumentsChain
        prompt_template = """You are a helpful assistant. Use the following context to answer the question.

                Context:
                {context}

                Question:
                {question}

                Answer:
                
                """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        self.combine_docs_chain = StuffDocumentsChain(llm_chain=self.llm_chain,document_variable_name="context")

    def get_memory_for_user(self, user_id):
        if user_id not in self.user_memory:
            self.user_memory[user_id] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
        return self.user_memory[user_id]

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
        self.vectorstore = Chroma.from_texts(
            all_texts, self.embeddings, persist_directory=self.persist_dir
        )
        self.vectorstore.persist()
        print("[INFO] Vectorstore created and persisted.")

    def load_vectorstore(self):
        if os.path.exists(self.persist_dir):
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir, embedding_function=self.embeddings
            )
            print("[INFO] Vectorstore loaded from disk.")
        else:
            print("[WARN] Vectorstore directory not found. Run load_and_embed_pdfs() first.")

    def ask(self, user_id: str, query: str, top_k=2, distance_threshold=0.25):
        if not self.vectorstore:
            return "Vectorstore not loaded. Run load_and_embed_pdfs() first."

        # Step 1: RAG retrieval
        pdf_docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        pdf_docs = [doc for doc, _ in pdf_docs_with_scores] if pdf_docs_with_scores else []

        tavily_retriever = TavilySearch(topic="general")
        web_docs_raw = tavily_retriever.invoke({"query": query, "num_results": top_k}) if pdf_docs else []
        web_docs = []  # extract text from Tavily dict
        for d in web_docs_raw:
            if isinstance(d, dict):
                web_docs.append(d.get("snippet", ""))
            elif isinstance(d, str):
                web_docs.append(d)
                
        combined_texts = [getattr(d, "page_content", "") for d in pdf_docs] + web_docs
        if not combined_texts:
            combined_texts = [query]  # fallback

        # Convert to Document objects
        documents = [Document(page_content=text) for text in combined_texts]

        # Step 2: Memory
        memory = self.get_memory_for_user(user_id)
        history = "\n".join([f"{m.role}: {m.content}" for m in memory.load_memory_variables({})["chat_history"]])
        question_with_history = f"{history}\n\n{query}" if history else query

        # Step 3: Run the chain
        try:
            result = self.combine_docs_chain.run({
                "input_documents": documents,
                "question": question_with_history
            })
            # Step 4: Update memory
            memory.save_context({"role": "user", "content": query}, {"role": "assistant", "content": result})
            return result
        except Exception as e:
            return f"Error talking to model: {e}"
