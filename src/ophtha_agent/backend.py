from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import json
import math
import re
from typing import Protocol

from .schemas import RetrievedEvidence


class RAGBackend(Protocol):
    """Small contract that lets the Agent reuse any existing retriever."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]: ...

    def rerank(
        self, query: str, documents: list[RetrievedEvidence]
    ) -> list[RetrievedEvidence]: ...


class MockRAGBackend:
    """Deterministic backend for CPU tests and interface smoke checks."""

    def __init__(self, documents: Iterable[RetrievedEvidence] | None = None):
        self.documents = list(documents or self._default_documents())
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _default_documents() -> list[RetrievedEvidence]:
        return [
            RetrievedEvidence(
                doc_id="eye-001",
                text="干眼通常与泪液不足或泪液蒸发过快有关，常见表现包括眼涩、异物感和视疲劳。",
                score=0.9,
                source="mock_ophthalmology_guideline",
            ),
            RetrievedEvidence(
                doc_id="eye-002",
                text="若出现突发视力下降、明显眼痛、畏光或外伤，应尽快到眼科就诊。",
                score=0.8,
                source="mock_urgent_care_guideline",
            ),
            RetrievedEvidence(
                doc_id="eye-003",
                text="眼底检查结果应结合眼科医生的临床检查，不应仅凭线上问答自行诊断或停药。",
                score=0.7,
                source="mock_safety_guideline",
            ),
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        self.calls.append(("retrieve", query))
        terms = {term for term in query.lower().split() if len(term) > 1}
        ranked = sorted(
            self.documents,
            key=lambda doc: sum(term in doc.text.lower() for term in terms),
            reverse=True,
        )
        return ranked[:top_k]

    def rerank(
        self, query: str, documents: list[RetrievedEvidence]
    ) -> list[RetrievedEvidence]:
        self.calls.append(("rerank", query))
        return sorted(documents, key=lambda doc: doc.score, reverse=True)


class JsonlCorpusBackend:
    """CPU-friendly retrieval backend for the upstream public QA corpus.

    This deterministic lexical fallback makes the integrated project runnable
    before heavyweight FAISS and reranker assets are installed. When the
    original project's assets are available, use LegacyFundusBackend instead.
    """

    _token_pattern = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

    def __init__(self, documents: Iterable[RetrievedEvidence]):
        self.documents = list(documents)
        self._tokens = [self._tokenize(doc.text) for doc in self.documents]
        self._idf = self._build_idf(self._tokens)
        self.calls: list[tuple[str, str]] = []

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        max_docs: int | None = None,
        include_answers: bool = False,
    ) -> "JsonlCorpusBackend":
        documents: list[RetrievedEvidence] = []
        with open(path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if max_docs is not None and index >= max_docs:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                text = cls._row_text(row, include_answer=include_answers)
                if not text:
                    continue
                documents.append(
                    RetrievedEvidence(
                        doc_id=f"corpus-{len(documents):06d}",
                        text=text,
                        source=str(row.get("source", "public_qa_corpus")),
                    )
                )
        if not documents:
            raise ValueError(f"no usable JSONL records found in {path}")
        return cls(documents)

    @staticmethod
    def _row_text(row: dict, include_answer: bool = False) -> str:
        question = row.get("question") or row.get("input") or row.get("prompt") or ""
        context = row.get("context") or row.get("instruction") or ""
        answer = row.get("answer") or row.get("output") or ""
        parts = (question, context, answer) if include_answer else (question, context)
        return "\n".join(str(part).strip() for part in parts if str(part).strip())

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [token.lower() for token in cls._token_pattern.findall(text)]

    @staticmethod
    def _build_idf(token_lists: list[list[str]]) -> dict[str, float]:
        df: defaultdict[str, int] = defaultdict(int)
        for tokens in token_lists:
            for token in set(tokens):
                df[token] += 1
        count = max(1, len(token_lists))
        return {token: math.log((count + 1) / (freq + 1)) + 1.0 for token, freq in df.items()}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        self.calls.append(("retrieve", query))
        query_set = set(self._tokenize(query))
        if not query_set:
            return self.documents[:top_k]
        scored: list[tuple[float, RetrievedEvidence]] = []
        for doc, tokens in zip(self.documents, self._tokens):
            overlap = query_set & set(tokens)
            score = sum(self._idf.get(token, 1.0) for token in overlap)
            if score:
                scored.append((score, doc.model_copy(update={"score": score})))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]] or self.documents[:top_k]

    def rerank(
        self, query: str, documents: list[RetrievedEvidence]
    ) -> list[RetrievedEvidence]:
        self.calls.append(("rerank", query))
        query_set = set(self._tokenize(query))
        return sorted(
            documents,
            key=lambda doc: (len(query_set & set(self._tokenize(doc.text))), doc.score),
            reverse=True,
        )
