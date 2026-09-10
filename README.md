# Ophtha-Agent

一个可以直接挂在现有眼底病 RAG 项目上的 Agent 外壳。

GitHub: https://github.com/qqnnhhdmpc666/ophtha-agent

原项目的 RAG、QLoRA、Type-Aware 和 Evidence Traceability 结果保留；本目录只增加状态化 Agent、证据验证和可回放轨迹，不扩展 DPO、ORPO、GRPO 或其他训练流程。简历升级稿见 `docs/RESUME_UPGRADE.md`。

本目录不复制或破坏桌面上的原项目，而是通过适配器接入它的检索能力：

```text
LangGraph 工作流
    ↓
FAISS / BM25 / Cross-Encoder 检索适配器
    ↓
Qwen 或其他 OpenAI-compatible 模型
    ↓
Evidence Verifier
    ↓
带引用回答 / 重检索 / 安全拒答
```

## 当前已完成

- 明确的 Agent 状态机：classify → retrieve → rerank → draft → verify → retry/finish；
- `RAGBackend` 接口，可接入现有眼底 RAG、Neo4j 或 Mock 后端；
- Pydantic 结构化输入输出；
- Evidence Verifier，拒绝没有有效证据引用的回答；
- 单次重检索上限，避免小模型循环；
- JSONL 轨迹记录和 no-overwrite 检查；
- 公共 QA JSONL 的答案隔离读取器和 CPU 词法检索 fallback；
- 统一评测摘要和可选 FastAPI `/health`、`/query` 接口；
- CPU-only Mock 端到端测试；
- 可选 LangGraph 实现，未安装 LangGraph 时仍可运行确定性 fallback。

## CPU Mock 验证

```powershell
cd "C:\Users\31552\Documents\my enhance project\ophtha-agent"
python -m pytest -q
python -m ophtha_agent.cli --mock
```

启动 CPU Mock HTTP 服务：

```powershell
python -m uvicorn scripts.run_mock_api:app --reload
```

使用上游公开 QA JSONL 做无答案泄漏的 CPU 检索 smoke：

```powershell
python -m ophtha_agent.cli --mock `
  --corpus data\smoke_qa.jsonl
```

默认不会把 JSONL 的 `answer/output` 字段放进检索证据。公共 QA 文件可以用于数据格式和训练候选审计，不能直接当作经过临床审核的知识库。上游数据接入说明见 `docs/UPSTREAM_DATA.md`。

## 接入现有眼底 RAG

现有仓库的 `qa_system.py` 是一个紧耦合的线性 RAG 流程，因此不要让 Agent 直接调用它的最终回答函数。应当为它实现只返回文档和分数的检索适配器：

```python
from ophtha_agent.adapters.legacy_fundus import LegacyFundusBackend
from ophtha_agent.agent import OphthaAgent

backend = LegacyFundusBackend.from_module_path(
    r"D:\path\to\low-resource-fundus-qa"
)
agent = OphthaAgent(backend=backend)
result = agent.run("什么是干眼？")
```

适配器支持现有项目常见的 `retrieve`、`search`、`hybrid_search` 和 `retrieve_documents` 命名；如果原项目只有最终回答函数，会明确报错，不会假装生成了可验证证据。

## 与桌面项目的分工

- 智能客服项目：可复用 LangGraph、API、状态管理和多工具编排思路；
- 眼底 RAG：提供真正的医疗知识检索后端；
- ChatGLM-RecAlign：不接入本 Agent 运行时；原项目的 QLoRA 结果只作为既有 baseline；
- Finance/PPT/数学 GRPO：暂不混入医疗 Agent 主线。

当前版本只验证 Agent 工程链路。真实模型、真实 FAISS 索引和医学效果对比仍需单独运行，不能用 Mock 结果代替。
