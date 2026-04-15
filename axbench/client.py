import time
import httpx


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        request_timeout_s: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.request_timeout_s = request_timeout_s

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[str, float]:
        """Returns (response_text, latency_ms)."""
        start = time.monotonic()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=httpx.Timeout(10.0, read=self.request_timeout_s),
            )
        except httpx.TimeoutException as exc:
            raise ValueError(
                f"Request to {self.base_url}/chat/completions timed out after "
                f"{self.request_timeout_s:.0f}s while waiting for model {self.model!r}."
            ) from exc
        self._raise_for_status(response)
        latency_ms = (time.monotonic() - start) * 1000
        message = response.json()["choices"][0]["message"]
        text = self._extract_message_text(message)
        return text, latency_ms

    def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
    ) -> tuple[dict, float]:
        """For tool calling evaluation (Pillar 5). Returns (response_dict, latency_ms)."""
        start = time.monotonic()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "temperature": temperature,
                },
                timeout=httpx.Timeout(10.0, read=self.request_timeout_s),
            )
        except httpx.TimeoutException as exc:
            raise ValueError(
                f"Tool-calling request to {self.base_url}/chat/completions timed out after "
                f"{self.request_timeout_s:.0f}s while waiting for model {self.model!r}."
            ) from exc
        self._raise_for_status(response)
        latency_ms = (time.monotonic() - start) * 1000
        return response.json()["choices"][0]["message"], latency_ms

    def _extract_message_text(self, message: dict) -> str:
        # Use the first non-empty field. Reasoning fields (reasoning_content,
        # reasoning) are internal chain-of-thought; content is the final answer.
        # Concatenating them causes thinking text to leak into code responses.
        for key in ("content", "reasoning_content", "reasoning"):
            text = self._normalize_text_content(message.get(key))
            if text:
                return text
        return ""

    def _normalize_text_content(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            normalized_parts = []
            for item in value:
                if isinstance(item, str):
                    normalized_parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        normalized_parts.append(text)
            return "\n".join(part for part in normalized_parts if part).strip()
        return str(value)

    def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_message(response)
            if response.status_code == 404 and "model" in detail.lower() and "does not exist" in detail.lower():
                available_models = self._get_available_models()
                available_suffix = ""
                if available_models:
                    available_suffix = f" Available model ids: {', '.join(available_models)}."
                raise ValueError(
                    f"Model {self.model!r} was not found at {self.base_url}.{available_suffix} "
                    f"Server message: {detail}"
                ) from exc
            raise ValueError(
                f"Request to {response.request.url} failed with status {response.status_code}. "
                f"Server message: {detail}"
            ) from exc

    def _extract_error_message(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except Exception:
            return response.text.strip() or response.reason_phrase

        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
        if isinstance(error, str):
            return error
        return response.text.strip() or response.reason_phrase

    def _get_available_models(self) -> list[str]:
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        models = []
        for entry in payload.get("data", []):
            model_id = entry.get("id")
            if model_id:
                models.append(str(model_id))
        return models
