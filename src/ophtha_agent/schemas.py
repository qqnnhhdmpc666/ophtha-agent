from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievedEvidence(BaseModel):
    doc_id: str
    text: str
    score: float = 0.0
    source: str | None = None


class VerificationResult(BaseModel):
    supported: bool
    citation_ids: list[str] = Field(default_factory=list)
    valid_citation_ids: list[str] = Field(default_factory=list)
    support_score: float = 0.0
    reason: str


class AgentResponse(BaseModel):
    request_id: str
    status: Literal["answered", "abstained", "error"]
    answer: str
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    verification: VerificationResult
    retrieval_attempts: int = 0
    risk_level: Literal["normal", "urgent", "emergency"] = "normal"
    trace_path: str | None = None


def dump_model(value: BaseModel | Any) -> dict[str, Any]:
    """Support Pydantic v2 and v1 during local migration."""

    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()

