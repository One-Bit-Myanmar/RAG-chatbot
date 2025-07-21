from fastapi import FastAPI
from pydantic import BaseModel
import main

app = FastAPI()

API_KEY = "API KEY"
TAVILY_API_KEY = "Search Key"
rag = main.RAG(api_key=API_KEY,search_key=TAVILY_API_KEY)
# rag.load_and_embed_pdfs() #only for first time and add new pdf

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(req: QuestionRequest):    
    try:
        answer = rag.ask(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        return {"error": str(e)}
