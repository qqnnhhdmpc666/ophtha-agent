# Agent Overlay Design

## Positioning

The Agent is an orchestration layer over the existing ophthalmology RAG system. It does not replace the domain retriever, claim a new optimizer, or introduce a new training stage.

```text
Existing core: Qwen + QLoRA + FAISS/BM25/Cross-Encoder + Type-Aware RAG
                         ↑
New overlay: LangGraph state machine + typed tools + evidence gate + trace
```

## State machine

```text
START
  → classify/risk triage
  → retrieve
  → rerank
  → draft answer
  → verify evidence
       ├─ supported → ANSWER
       ├─ unsupported and retry budget available → rewrite query → retrieve
       └─ unsupported and budget exhausted → ABSTAIN
```

The retry budget is explicit and bounded. A model cannot silently bypass verification by calling the old final-answer function.

## Tool boundaries

The Agent only depends on this small contract:

```text
retrieve(query, top_k) -> Evidence[]
rerank(query, Evidence[]) -> Evidence[]
draft(question, Evidence[]) -> answer
verify(answer, Evidence[]) -> VerificationResult
```

This allows the same Agent workflow to use the existing FAISS/BM25/Cross-Encoder implementation, a Neo4j-backed retriever, or a CPU Mock backend.

## Why this is a mature Agent design

- State is explicit instead of hidden in prompt text;
- tool input/output is typed and normalized;
- evidence verification is outside the language model;
- retry and termination conditions are deterministic;
- every retrieval, verification, retry and final status is traceable;
- the original RAG baseline remains available for causal comparison.

## Evaluation boundary

The primary project result remains the original fixed RAG/QLoRA evaluation. The Agent overlay adds a second comparison condition:

```text
fixed RAG baseline vs Agent-RAG overlay
```

Report answer quality, evidence support, citation validity, safe abstention, latency and retrieval-call count. Do not report Mock results as clinical improvement.

