"""Ophtha-Agent: a thin, testable Agent layer for an existing RAG backend."""

from .agent import OphthaAgent
from .backend import MockRAGBackend, RAGBackend
from .model import MockAnswerModel
from .schemas import AgentResponse, RetrievedEvidence
from .verifier import EvidenceVerifier

__all__ = [
    "AgentResponse",
    "EvidenceVerifier",
    "MockAnswerModel",
    "MockRAGBackend",
    "OphthaAgent",
    "RAGBackend",
    "RetrievedEvidence",
]

