"""Ophtha-Agent: a thin, testable Agent layer for an existing RAG backend."""

from .agent import OphthaAgent
from .backend import JsonlCorpusBackend, MockRAGBackend, RAGBackend
from .evaluation import evaluate_questions
from .model import MockAnswerModel
from .schemas import AgentResponse, RetrievedEvidence
from .verifier import EvidenceVerifier

__all__ = [
    "AgentResponse",
    "EvidenceVerifier",
    "evaluate_questions",
    "JsonlCorpusBackend",
    "MockAnswerModel",
    "MockRAGBackend",
    "OphthaAgent",
    "RAGBackend",
    "RetrievedEvidence",
]
