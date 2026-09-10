from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from ..schemas import RetrievedEvidence


class LegacyFundusBackend:
    """Duck-typed adapter for the existing low-resource-fundus-qa repository."""

    SEARCH_METHODS = (
        "retrieve",
        "search",
        "hybrid_search_with_scores",
        "hybrid_search",
        "retrieve_documents",
    )

    def __init__(self, system: Any):
        self.system = system

    @classmethod
    def from_module_path(
        cls,
        repo_path: str | Path,
        module_name: str = "qa_system",
        factory_name: str = "get_eye_qa_system",
        system_kwargs: dict[str, Any] | None = None,
    ) -> "LegacyFundusBackend":
        repo_path = str(Path(repo_path).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        module = importlib.import_module(module_name)
        # The current upstream helper returns ``.answer`` (a callable), not
        # the system object. Prefer the real class so retrieval-only methods
        # remain available to the Agent.
        system_class = getattr(module, "EyeQASystem", None)
        if system_class is not None:
            return cls(system_class(**(system_kwargs or {})))
        factory = getattr(module, factory_name, None)
        if factory is None:
            raise AttributeError(f"{module_name}.EyeQASystem and {module_name}.{factory_name} were not found")
        system = factory()
        if callable(system) and not any(callable(getattr(system, name, None)) for name in cls.SEARCH_METHODS):
            raise RuntimeError(
                f"{module_name}.{factory_name} returned a final answer callable, not a retrieval backend"
            )
        return cls(system)

    def _call_search(self, query: str, top_k: int) -> Any:
        for name in self.SEARCH_METHODS:
            method = getattr(self.system, name, None)
            if not callable(method):
                continue
            for kwargs in ({"top_k": top_k}, {"k": top_k}, {}):
                try:
                    return method(query, **kwargs)
                except TypeError:
                    continue
        raise RuntimeError(
            "The legacy system exposes no retrieval backend or retrieval-only method. "
            "Add a small adapter around its FAISS/BM25/Cross-Encoder search path; "
            "do not fallback to answer_with_retrieval()."
        )

    @staticmethod
    def _normalize(item: Any, index: int) -> RetrievedEvidence:
        if isinstance(item, RetrievedEvidence):
            return item
        if isinstance(item, tuple) and len(item) >= 2:
            return RetrievedEvidence(
                doc_id=f"legacy-{index}",
                text=str(item[0]),
                score=float(item[1]),
                source="legacy",
            )
        if isinstance(item, dict):
            doc_id = item.get("doc_id") or item.get("id") or item.get("source") or f"legacy-{index}"
            text = item.get("text") or item.get("content") or item.get("page_content") or str(item)
            score = item.get("score") or item.get("similarity") or 0.0
            return RetrievedEvidence(doc_id=str(doc_id), text=str(text), score=float(score), source=str(item.get("source", "legacy")))
        text = getattr(item, "page_content", None) or getattr(item, "text", None) or str(item)
        doc_id = getattr(item, "doc_id", None) or getattr(item, "id", None) or f"legacy-{index}"
        score = getattr(item, "score", 0.0)
        return RetrievedEvidence(doc_id=str(doc_id), text=str(text), score=float(score), source="legacy")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedEvidence]:
        raw = self._call_search(query, top_k)
        if isinstance(raw, dict):
            raw = raw.get("documents") or raw.get("retrieved_docs") or raw.get("results") or []
        return [self._normalize(item, i) for i, item in enumerate(raw)][:top_k]

    def rerank(self, query: str, documents: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
        method = getattr(self.system, "rerank", None) or getattr(self.system, "rerank_documents", None)
        if callable(method):
            try:
                raw = method(query, documents)
                return [self._normalize(item, i) for i, item in enumerate(raw)]
            except (TypeError, AttributeError):
                pass
        return sorted(documents, key=lambda item: item.score, reverse=True)
