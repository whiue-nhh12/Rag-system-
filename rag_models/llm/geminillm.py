import json
import os
from typing import Optional
from urllib import error, request

from sympy import re

from .basellm import BaseLLM


class GeminiLLM(BaseLLM):
    """Simple Gemini client for sending a prompt and getting a text answer."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        super().__init__(model_name=model_name)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def ask(self, query: str) -> str:
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY. Set it in your environment or pass api_key.")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": query}
                    ]
                }
            ]
        }

        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini API request failed: {detail}") from exc

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini API response: {data}") from exc

    def hyde_generate(self,query : str) -> str:
        content = f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        result =  self.ask(content)
        return result

