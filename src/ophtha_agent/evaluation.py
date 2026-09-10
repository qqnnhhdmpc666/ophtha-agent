from __future__ import annotations

import time
from typing import Iterable

from .agent import OphthaAgent


def evaluate_questions(agent: OphthaAgent, questions: Iterable[str]) -> dict:
    """Run a deterministic, model-agnostic smoke/evaluation summary."""

    records = []
    for index, question in enumerate(questions):
        started = time.perf_counter()
        result = agent.run(question, request_id=f"eval-{index:04d}")
        records.append(
            {
                "id": f"eval-{index:04d}",
                "status": result.status,
                "supported": result.verification.supported,
                "support_score": result.verification.support_score,
                "retrieval_attempts": result.retrieval_attempts,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
    count = len(records)
    return {
        "count": count,
        "answered_rate": sum(item["status"] == "answered" for item in records) / max(1, count),
        "evidence_supported_rate": sum(item["supported"] for item in records) / max(1, count),
        "mean_latency_ms": round(sum(item["latency_ms"] for item in records) / max(1, count), 3),
        "mean_retrieval_attempts": round(sum(item["retrieval_attempts"] for item in records) / max(1, count), 3),
        "records": records,
    }

