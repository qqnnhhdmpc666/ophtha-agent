from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from .backend import RAGBackend
from .model import AnswerModel, MockAnswerModel
from .schemas import AgentResponse, dump_model
from .tracing import TraceWriter
from .verifier import EvidenceVerifier


class OphthaAgent:
    """Fixed, bounded Agent loop around an existing retrieval backend."""

    def __init__(
        self,
        backend: RAGBackend,
        model: AnswerModel | None = None,
        verifier: EvidenceVerifier | None = None,
        max_retries: int = 1,
        trace_dir: str | Path | None = None,
    ):
        self.backend = backend
        self.model = model or MockAnswerModel()
        self.verifier = verifier or EvidenceVerifier()
        self.max_retries = max(0, max_retries)
        self.trace_dir = Path(trace_dir) if trace_dir else None

    @staticmethod
    def _risk_level(question: str) -> str:
        urgent_terms = ("突然失明", "视力骤降", "剧烈眼痛", "眼外伤")
        emergency_terms = ("完全看不见", "化学品入眼")
        if any(term in question for term in emergency_terms):
            return "emergency"
        if any(term in question for term in urgent_terms):
            return "urgent"
        return "normal"

    @staticmethod
    def _rewrite_query(question: str, attempt: int) -> str:
        return f"{question} 眼科指南 症状 风险处理 证据检索第{attempt}次"

    def run(self, question: str, request_id: str | None = None) -> AgentResponse:
        if not question.strip():
            raise ValueError("question must not be empty")
        request_id = request_id or uuid.uuid4().hex
        trace_path = None
        writer = None
        if self.trace_dir:
            trace_path = self.trace_dir / f"{request_id}.jsonl"
            writer = TraceWriter(trace_path)

        def record(event: str, **payload: object) -> None:
            if writer:
                writer.write(event, {"request_id": request_id, **payload})

        risk_level = self._risk_level(question)
        record("episode_start", question_hash=hashlib.sha256(question.encode()).hexdigest(), risk_level=risk_level)
        last_evidence = []
        last_verification = None
        answer = ""
        attempts = 0

        for attempts in range(self.max_retries + 1):
            query = question if attempts == 0 else self._rewrite_query(question, attempts)
            record("retrieve_start", attempt=attempts, query_hash=hashlib.sha256(query.encode()).hexdigest())
            evidence = self.backend.retrieve(query, top_k=5)
            evidence = self.backend.rerank(query, evidence)
            record("retrieval_complete", attempt=attempts, evidence_ids=[doc.doc_id for doc in evidence])
            answer = self.model.draft(question, evidence)
            verification = self.verifier.verify(answer, evidence)
            record("verification", attempt=attempts, result=dump_model(verification))
            last_evidence = evidence
            last_verification = verification
            if verification.supported:
                record("episode_end", status="answered", attempts=attempts + 1)
                return AgentResponse(
                    request_id=request_id,
                    status="answered",
                    answer=answer,
                    evidence=evidence,
                    verification=verification,
                    retrieval_attempts=attempts + 1,
                    risk_level=risk_level,
                    trace_path=str(trace_path) if trace_path else None,
                )
            record("retry", attempt=attempts)

        assert last_verification is not None
        answer = self.verifier.abstention(question)
        record("episode_end", status="abstained", attempts=attempts + 1)
        return AgentResponse(
            request_id=request_id,
            status="abstained",
            answer=answer,
            evidence=last_evidence,
            verification=last_verification,
            retrieval_attempts=attempts + 1,
            risk_level=risk_level,
            trace_path=str(trace_path) if trace_path else None,
        )

