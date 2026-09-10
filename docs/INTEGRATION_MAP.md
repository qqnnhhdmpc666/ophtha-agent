# Integration Map

## Decision

The new project is a composition layer, not a destructive merge:

| Existing asset | Reuse | Boundary |
|---|---|---|
| `居丽叶简历项目1：智能客服Agent` | LangGraph state/workflow patterns, API and tool orchestration | Do not copy its `.env`, logs, database data or frontend build artifacts |
| `low-resource-fundus-qa` | FAISS/BM25/Cross-Encoder retrieval and ophthalmology corpus | Expose retrieval-only methods through `LegacyFundusBackend` |
| `ChatGLM-RecAlign` | Historical QLoRA/recommendation reference only | Not included in this Agent runtime |
| `Finance/Financial-MCP-Agent` | MCP tool wrapping pattern | Do not import financial prompts or claims into medical flow |
| PPT and math GRPO projects | None in v0 | Different action space and verifier |

## Upstream mismatch found

The current ophthalmology upstream exposes `EyeQASystem.hybrid_search_with_scores`, but its convenience function `get_eye_qa_system()` returns the final `.answer` callable. The adapter therefore prefers `EyeQASystem` and explicitly rejects a final-answer-only callable. This prevents an unverified answer from being mistaken for retrieved evidence.

The upstream `EyeQASystem` also loads its own generation model during initialization. For the first real integration, either supply the existing model/LoRA/RAG paths or add a retrieval-only constructor in the upstream copy. The new Agent does not silently load a second model or silently fall back to final-answer generation.

## v0 acceptance

- Mock workflow: passed;
- LangGraph compilation: passed;
- no-overwrite trace: passed;
- citation gate and bounded retry: passed;
- DPO/ORPO/GRPO: deliberately out of scope;
- real medical effectiveness: not checked offline;
- real upstream FAISS/Reranker assets: not checked because the cloned repository has no committed model checkpoint or built FAISS directory.
