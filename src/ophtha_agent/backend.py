from __future__ import annotations

from collections.abc import Iterable
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

