from __future__ import annotations

import argparse
import json

from .agent import OphthaAgent
from .backend import JsonlCorpusBackend, MockRAGBackend
from .model import OpenAICompatibleAnswerModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ophtha-Agent integration shell")
    parser.add_argument("question", nargs="?", default="什么是干眼？")
    parser.add_argument("--mock", action="store_true", help="use deterministic CPU backend/model")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=None, help="remote/local model name")
    parser.add_argument("--corpus", default=None, help="JSONL QA corpus for CPU lexical retrieval")
    parser.add_argument("--max-docs", type=int, default=None, help="limit corpus records for a smoke run")
    parser.add_argument("--trace-dir", default="runs/traces")
    args = parser.parse_args()
    backend = (
        JsonlCorpusBackend.from_jsonl(args.corpus, max_docs=args.max_docs)
        if args.corpus
        else MockRAGBackend()
    )
    if args.mock or not args.base_url or not args.model:
        agent = OphthaAgent(backend, trace_dir=args.trace_dir)
    else:
        model = OpenAICompatibleAnswerModel(args.base_url, args.model)
        agent = OphthaAgent(backend, model=model, trace_dir=args.trace_dir)
    result = agent.run(args.question)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
