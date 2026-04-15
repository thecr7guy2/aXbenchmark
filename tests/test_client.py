import json
import httpx
import pytest
from pytest_httpx import HTTPXMock
from axbench.client import LLMClient


def test_generate_returns_text_and_latency(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        json={"choices": [{"message": {"content": "hello world"}}]}
    )
    client = LLMClient("http://localhost:8000/v1", "test-model")
    text, latency = client.generate([{"role": "user", "content": "hi"}])
    assert text == "hello world"
    assert latency >= 0.0


def test_generate_sends_correct_payload(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        json={"choices": [{"message": {"content": "ok"}}]}
    )
    client = LLMClient("http://localhost:8000/v1", "my-model", api_key="EMPTY")
    client.generate([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=512)
    request = httpx_mock.get_requests()[0]
    body = request.read()
    payload = json.loads(body)
    assert payload["model"] == "my-model"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 512


def test_generate_raises_on_http_error(httpx_mock: HTTPXMock):
    httpx_mock.add_response(status_code=500)
    client = LLMClient("http://localhost:8000/v1", "test-model")
    with pytest.raises(ValueError, match="Request to"):
        client.generate([{"role": "user", "content": "hi"}])


def test_generate_reports_available_models_on_model_not_found(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        status_code=404,
        json={
            "error": {
                "message": "The model `wrong-model` does not exist.",
                "type": "NotFoundError",
                "code": 404,
            }
        },
    )
    httpx_mock.add_response(
        json={
            "data": [
                {"id": "minimax-m2.5-awq"},
            ]
        }
    )
    client = LLMClient("http://localhost:8000/v1", "wrong-model")
    with pytest.raises(ValueError, match="Available model ids: minimax-m2.5-awq"):
        client.generate([{"role": "user", "content": "hi"}])


def test_generate_reports_timeout_cleanly(httpx_mock: HTTPXMock):
    httpx_mock.add_exception(httpx.ReadTimeout("timed out"))
    client = LLMClient("http://localhost:8000/v1", "test-model", request_timeout_s=5.0)
    with pytest.raises(ValueError, match="timed out after 5s"):
        client.generate([{"role": "user", "content": "hi"}])


def test_generate_handles_null_content_with_reasoning_text(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        json={
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "```python\ndef add(a, b):\n    return a + b\n```",
                    }
                }
            ]
        }
    )
    client = LLMClient("http://localhost:8000/v1", "test-model")
    text, _ = client.generate([{"role": "user", "content": "hi"}])
    assert "def add" in text
