from __future__ import annotations

import json
import urllib.request
from typing import Protocol

from .schemas import RetrievedEvidence


class AnswerModel(Protocol):
    def draft(self, question: str, evidence: list[RetrievedEvidence]) -> str: ...


class MockAnswerModel:
    def draft(self, question: str, evidence: list[RetrievedEvidence]) -> str:
        if not evidence:
            return "目前检索不到足够依据，无法可靠回答。"
        first = evidence[0]
        return f"根据检索到的资料，{first.text} 如有明显或持续症状，请结合眼科医生检查。[{first.doc_id}]"


class OpenAICompatibleAnswerModel:
    """Minimal local-server client; usable with vLLM/Ollama-compatible APIs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout

    def draft(self, question: str, evidence: list[RetrievedEvidence]) -> str:
        evidence_text = "\n".join(
            f"[{doc.doc_id}] {doc.text}" for doc in evidence
        )
        prompt = (
            "你是眼科知识问答助手。只能根据给出的证据回答，不能自行补充医学事实。"
            "回答末尾必须引用使用过的证据ID，例如 [eye-001]。若证据不足，明确说明无法判断。\n\n"
            f"问题：{question}\n证据：\n{evidence_text}"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "遵守证据约束，输出简洁安全的中文回答。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 256,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]
