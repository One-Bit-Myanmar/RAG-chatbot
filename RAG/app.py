from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator
import json
import httpx
import main  # your RAG class

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate RAG
rag = main.RAG(search_key = "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")
user_sessions = {}  # user_id -> memory handled inside RAG

# Pydantic model for requests
class QuestionRequest(BaseModel):
    user_id: str
    question: str


def get_session(user_id: str):
    """
    Ensure each user has a separate memory session in RAG.
    """
    return user_id  # memory handled inside RAG.ask(user_id, query)


@app.post("/ask/")
async def ask_question(req: QuestionRequest):
    try:
        answer = rag.ask(req.user_id, req.question)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.post("/ask/local/streaming")
async def ask_question_streaming(req: QuestionRequest):
    try:
        ollama_url, prompt = await rag.ask_deepseek_local_async(req.question)

        return StreamingResponse(
            response_generator(ollama_url, prompt),
            media_type="application/json"
        )
    except Exception as e:
        return {"error": str(e)}


@app.post("/ask/local")
async def ask_question_local(req: QuestionRequest):
    try:
        answer = rag.ask_deepseek_local(req.question)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Streaming helpers
async def ollama_stream(ollama_url: str, prompt: str):
    """
    Async generator to stream lines from Ollama local LLM.
    """
    url = f"{ollama_url}/api/chat"
    payload = {
        "model": "deepseek-r1:7b",
        "stream": True,
        "messages": [{"role": "user", "content": prompt.strip()}]
    }

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload, timeout=60) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line


async def response_generator(ollama_url: str, prompt: str) -> AsyncGenerator[dict, None]:
    """
    Parse streaming lines from Ollama and yield JSON blocks.
    """
    async for line in ollama_stream(ollama_url, prompt):
        try:
            block = json.loads(line)
        except json.JSONDecodeError:
            continue
        if block:
            yield json.dumps(block)
        if block.get("done", False):
            break
