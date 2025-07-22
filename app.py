from fastapi import FastAPI
from pydantic import BaseModel
import main
import asyncio

app = FastAPI()

API_KEY = "API Key"
TAVILY_API_KEY = "Search key"

rag = main.RAG(api_key=API_KEY, search_key=TAVILY_API_KEY)

class QuestionRequest(BaseModel):
    question: str

# @app.on_event("startup")
# async def startup_event():
#     # Run this only if you want to update the embeddings on server start
#     # Comment out if embeddings are already persisted
#     # You can also run this manually outside the server
#     await asyncio.to_thread(rag.load_and_embed_pdfs)

@app.post("/ask/")
async def ask_question(req: QuestionRequest):    
    try:
        gemini_answer = rag.ask(req.question)
        local_deepseek = rag.ask_deepseek_local(req.question)
        return {"question": req.question, 
                "gemini_answer": gemini_answer,
                "local_deepseek" :local_deepseek
        }
    except Exception as e:
        return {"error": str(e)}
