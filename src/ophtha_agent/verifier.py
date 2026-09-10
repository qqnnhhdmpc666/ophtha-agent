from __future__ import annotations

import re

from .schemas import RetrievedEvidence, VerificationResult


class EvidenceVerifier:
    """Deterministic citation gate for the first integration version.

    This is an engineering verifier, not a clinical diagnosis model. It checks
    that the answer cites retrieved document IDs; richer claim-level checking
    can be added after the backend contract is stable.
    """

    _citation_pattern = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")

    def verify(
        self, answer: str, evidence: list[RetrievedEvidence]
    ) -> VerificationResult:
        cited = self._citation_pattern.findall(answer)
        available = {doc.doc_id for doc in evidence}
        valid = [doc_id for doc_id in cited if doc_id in available]
        score = min(1.0, len(set(valid)) / max(1, len(set(cited)))) if cited else 0.0
        supported = bool(valid)
        reason = "至少一个引用来自当前检索证据" if supported else "回答没有引用当前检索证据"
        return VerificationResult(
            supported=supported,
            citation_ids=cited,
            valid_citation_ids=sorted(set(valid)),
            support_score=score,
            reason=reason,
        )

    @staticmethod
    def abstention(question: str) -> str:
        return (
            "根据当前知识库无法提供足够可靠的依据来回答这个问题。"
            "请补充检查报告或咨询眼科专业人员，不要仅凭线上回答自行诊断或调整用药。"
        )

