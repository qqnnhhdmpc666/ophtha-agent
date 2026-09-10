from __future__ import annotations

from typing import Any, TypedDict

from .backend import RAGBackend
from .model import AnswerModel
from .verifier import EvidenceVerifier


class GraphState(TypedDict, total=False):
    question: str
    query: str
    attempt: int
    max_retries: int
    evidence: list[Any]
    answer: str
    verification: dict[str, Any]
    status: str


def build_langgraph(
    backend: RAGBackend,
    model: AnswerModel,
    verifier: EvidenceVerifier | None = None,
    max_retries: int = 1,
):
    """Build the real LangGraph graph when the optional dependency is installed."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is optional for CPU mock mode. Install ophtha-agent[runtime] to build the graph."
        ) from exc

    verifier = verifier or EvidenceVerifier()

    def retrieve(state: GraphState) -> GraphState:
        query = state.get("query", state["question"])
        docs = backend.retrieve(query, top_k=5)
        return {"evidence": backend.rerank(query, docs)}

    def draft(state: GraphState) -> GraphState:
        return {"answer": model.draft(state["question"], state.get("evidence", []))}

    def verify(state: GraphState) -> GraphState:
        result = verifier.verify(state.get("answer", ""), state.get("evidence", []))
        return {"verification": result.model_dump()}

    def route(state: GraphState) -> str:
        if state.get("verification", {}).get("supported"):
            return "finish"
        if state.get("attempt", 0) >= state.get("max_retries", max_retries):
            return "finish"
        return "retry"

    def retry(state: GraphState) -> GraphState:
        next_attempt = state.get("attempt", 0) + 1
        return {
            "attempt": next_attempt,
            "query": f"{state['question']} 眼科指南 证据检索第{next_attempt}次",
        }

    def finish(state: GraphState) -> GraphState:
        if not state.get("verification", {}).get("supported"):
            return {"status": "abstained", "answer": verifier.abstention(state["question"])}
        return {"status": "answered"}

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("draft", draft)
    graph.add_node("verify", verify)
    graph.add_node("retry", retry)
    graph.add_node("finish", finish)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges("verify", route, {"retry": "retry", "finish": "finish"})
    graph.add_edge("retry", "retrieve")
    graph.add_edge("finish", END)
    return graph.compile()

