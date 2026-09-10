from __future__ import annotations

from .agent import OphthaAgent


def create_app(agent: OphthaAgent):
    """Create a small FastAPI surface for reuse by the existing web host."""

    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install ophtha-agent[api] to run the HTTP wrapper") from exc

    class QueryRequest(BaseModel):
        question: str = Field(min_length=1)
        request_id: str | None = None

    app = FastAPI(title="Ophtha-Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query")
    def query(request: QueryRequest):
        result = agent.run(request.question, request.request_id)
        return result.model_dump()

    return app

