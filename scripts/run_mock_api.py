from ophtha_agent.agent import OphthaAgent
from ophtha_agent.api import create_app
from ophtha_agent.backend import JsonlCorpusBackend


backend = JsonlCorpusBackend.from_jsonl("data/smoke_qa.jsonl")
app = create_app(OphthaAgent(backend, trace_dir="runs/api_traces"))

