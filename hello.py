import httpx
import json
    

async def ollama_stream(ollama_url, prompt):
    url = f"{ollama_url}/api/chat"
    payload = {
        "model": "deepseek-r1:7b",
        "stream": True,
        "messages": [{"role": "user", "content": prompt.strip()}]
    }
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload, timeout=60) as response:
            print(f"received response status_code={response.status_code}")
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line

async def response_generator(ollama_url, prompt):
    async for response in ollama_stream(ollama_url, prompt):
        try:
            block: dict = json.loads(response)
        except Exception:
            print("Error parsing JSON:")
            continue  # skip malformed lines
        
        if block:
            yield block

        if block.get("done", False):
            break
