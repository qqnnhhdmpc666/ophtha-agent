from pathlib import Path

import pytest

from ophtha_agent import EvidenceVerifier, MockAnswerModel, MockRAGBackend, OphthaAgent
from ophtha_agent.adapters.legacy_fundus import LegacyFundusBackend
from ophtha_agent.schemas import RetrievedEvidence


def test_mock_episode_answers_with_evidence(tmp_path: Path):
    result = OphthaAgent(MockRAGBackend(), trace_dir=tmp_path).run("什么是干眼？", "case-001")
    assert result.status == "answered"
    assert result.verification.supported is True
    assert result.retrieval_attempts == 1
    assert Path(result.trace_path).exists()


def test_trace_writer_is_no_overwrite(tmp_path: Path):
    agent = OphthaAgent(MockRAGBackend(), trace_dir=tmp_path)
    agent.run("什么是干眼？", "case-002")
    with pytest.raises(FileExistsError):
        agent.run("什么是干眼？", "case-002")


class NoCitationModel:
    def draft(self, question, evidence):
        return "我无法从证据中确定。"


def test_missing_citation_abstains_after_bounded_retry(tmp_path: Path):
    result = OphthaAgent(
        MockRAGBackend(), model=NoCitationModel(), max_retries=1, trace_dir=tmp_path
    ).run("一个没有足够证据的问题", "case-003")
    assert result.status == "abstained"
    assert result.retrieval_attempts == 2
    assert result.verification.supported is False


def test_verifier_rejects_unknown_citation():
    evidence = [RetrievedEvidence(doc_id="eye-001", text="事实", score=1.0)]
    result = EvidenceVerifier().verify("结论 [not-in-retrieval]", evidence)
    assert result.supported is False
    assert result.valid_citation_ids == []


def test_emergency_question_is_marked():
    result = OphthaAgent(MockRAGBackend()).run("化学品入眼怎么办？", "case-004")
    assert result.risk_level == "emergency"


class LegacySearchOnly:
    def hybrid_search_with_scores(self, query, k=5):
        return [("来自旧项目的证据", 0.88)]


def test_legacy_adapter_normalizes_scored_tuples():
    backend = LegacyFundusBackend(LegacySearchOnly())
    docs = backend.retrieve("眼底问题", top_k=1)
    assert docs[0].text == "来自旧项目的证据"
    assert docs[0].score == 0.88


class FinalAnswerOnly:
    def __call__(self, question):
        return "最终答案"


def test_legacy_adapter_does_not_treat_final_answer_as_retrieval():
    backend = LegacyFundusBackend(FinalAnswerOnly())
    with pytest.raises(RuntimeError, match="retrieval backend"):
        backend.retrieve("问题")


def test_langgraph_workflow_compiles_and_runs():
    pytest.importorskip("langgraph")
    from ophtha_agent.graph import build_langgraph

    graph = build_langgraph(MockRAGBackend(), MockAnswerModel())
    state = graph.invoke({"question": "什么是干眼？", "attempt": 0, "max_retries": 1})
    assert state["status"] == "answered"
    assert state["verification"]["supported"] is True
