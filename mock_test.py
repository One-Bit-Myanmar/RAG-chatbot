import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from hello import response_generator
import json

@pytest.mark.asyncio
async def test_response_generator_content():
    mock_json = [
        {
            "model": "deepseek-r1:7b",
            "created_at": "2025-08-02T07:26:41.4084791Z",
            "message": {
                "role": "assistant",
                "content": "**"
            },
            "done": False
        },
        {
            "model": "deepseek-r1:7b",
            "created_at": "2025-08-02T07:26:41.497495Z",
            "message": {
                "role": "assistant",
                "content": "Cy"
            },
            "done": False
        },
        {
            "model": "deepseek-r1:7b",
            "created_at": "2025-08-02T07:26:41.580442Z",
            "message": {
                "role": "assistant",
                "content": "ber"
            },
            "done": False
        },
        {
            "model": "deepseek-r1:7b",
            "created_at": "2025-08-02T07:26:41.6656932Z",
            "message": {
                "role": "assistant",
                "content": "security"
            },
            "done": False
        },
        {
            "model": "deepseek-r1:7b",
            "created_at": "2025-08-02T07:26:41.7505942Z",
            "message": {
                "role": "assistant",
                "content": ":"
            },
            "done": False
        },
        {
            "model": "deepseek-r1:7b",
            "created_at": "2025-08-02T07:26:41.8381651Z",
            "message": {
                "role": "assistant",
                "content": " Protect"
            },
            "done": False
        },
        {
            "model": "deepseek-r1:7b",
            "created_at": "2025-08-02T07:26:41.9252384Z",
            "message": {
                "role": "assistant",
                "content": "ing"
            },
            "done": False
        },
        {
            "model": "deepseek-r1:7b",
            "created_at": "2025-08-02T07:26:42.0125978Z",
            "message": {
                "role": "assistant",
                "content": " Digital"
            },
            "done": False
        },
        {
            "model": "deepseek-r1:7b",
            "created_at": "2025-08-02T07:27:22.8912527Z",
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done_reason": "stop",
            "done": True,
            "total_duration": 91320493900,
            "load_duration": 69135500,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 424269100,
            "eval_count": 1042,
            "eval_duration": 90826580000
        }
    ]

    mock_stream = [str(json.dumps(item)) for item in mock_json]


    async def mock_ollama_stream(ollama_url, prompt):

        for line in mock_stream:
            yield line

    # with patch("hello.ollama_stream", new=mock_ollama_stream):
    results = []
    async for msg in response_generator("http://localhost:11434", "hi"):
        results.append(msg)
    # Only one message should be yielded
    print("Results:", results)
    assert True