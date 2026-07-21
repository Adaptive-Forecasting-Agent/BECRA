"""OpenAI-compatible chat client for BECRA lesson induction / planning / verify."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

try:
    from .llm_local import API_KEY as _LOCAL_KEY
    from .llm_local import BASE_URL as _LOCAL_BASE
    from .llm_local import MODEL as _LOCAL_MODEL
except ImportError:
    _LOCAL_KEY = ""
    _LOCAL_BASE = ""
    _LOCAL_MODEL = ""

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _resolve_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    for env in ("BECRA_LLM_API_KEY", "OPENAI_API_KEY"):
        val = os.getenv(env, "").strip()
        if val:
            return val
    return (_LOCAL_KEY or "").strip()


def _parse_json_content(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for match in _JSON_FENCE.finditer(text):
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.api_key = _resolve_api_key(api_key)
        self.base_url = (
            base_url
            or os.getenv("BECRA_LLM_BASE_URL", "").strip()
            or (_LOCAL_BASE or "").strip()
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("BECRA_LLM_MODEL", "").strip()
            or (_LOCAL_MODEL or "").strip()
            or "gpt-4o"
        )
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, prompt: str, *, temperature: float = 0.2) -> str:
        if not self.available:
            raise RuntimeError("LLM API key not configured (BECRA_LLM_API_KEY / OPENAI_API_KEY / llm_local.py)")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": "You are a careful assistant. Follow the user format exactly."},
                {"role": "user", "content": prompt},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code == 429:
            raise RuntimeError(f"LLM rate limited (429): {resp.text[:500]}")
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    def complete_json(self, prompt: str, *, temperature: float = 0.2) -> dict[str, Any] | None:
        text = self.complete(prompt, temperature=temperature)
        return _parse_json_content(text)
