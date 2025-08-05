from fastapi import FastAPI
from pydantic import BaseModel
import RAG.main as main
from fastapi.responses import StreamingResponse
import json
import httpx 
from typing import AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()

API_KEY = "API Key"
TAVILY_API_KEY = "Search key"

rag = main.RAG(api_key=API_KEY, search_key=TAVILY_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # Set to True if your API uses cookies or authorization headers
    allow_methods=["*"],     # Or specify specific methods like ["GET", "POST"]
    allow_headers=["*"],     # Or specify specific headers
)

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
        # gemini_answer = rag.ask(req.question)
        local_deepseek = rag.ask_deepseek_local(req.question)
        return {"question": req.question, 
                # "gemini_answer": gemini_answer,
                "local_deepseek" :local_deepseek
        }
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/ask/local/streaming")
async def ask_question(req: QuestionRequest):    

    ollama_url, prompt = await rag.ask_deepseek_local_async(req.question)

    try: 
        return StreamingResponse(
            response_generator(ollama_url, prompt),
            media_type="application/json"
        )
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/ask/local")
async def ask_question_local(req: QuestionRequest):    
    try:
        localdeep_seek = rag.ask_deepseek_local(req.question)
        return {"response": localdeep_seek}
    except Exception as e:
        return {"error": str(e)}
    



async def ollama_stream(ollama_url, prompt):
    url = f"{ollama_url}/api/chat"
    payload = {
        "model": "deepseek-r1:7b",
        "stream": True,
        "messages": [{"role": "user", "content": prompt.strip()}]
    }
    
    async with httpx.AsyncClient() as client:
        print("[DEBUG] Created AsyncClient: ", client)
        async with client.stream("POST", url, json=payload, timeout=60) as response:
            print(f"[DEBUG] Received response={response}")
            response.raise_for_status()
            async for line in response.aiter_lines():
                print("[DEBUG] Received a line from stream: ", line)
                yield line

async def response_generator(ollama_url, prompt) -> AsyncGenerator[dict, None]:
    async for response in ollama_stream(ollama_url, prompt):
        print("[DEBUG] Received response from ollama_stream: ", response)
        try:
            block: dict = json.loads(response)
        except Exception:
            print("[DEBUG] Failed to parse JSON line")
            continue  # skip malformed lines
        
        if block:
            print("[DEBUG] Yielding block")
            yield json.dumps(block)

        if block.get("done", False):
            print("[DEBUG] Block indicates done, breaking loop")
            break