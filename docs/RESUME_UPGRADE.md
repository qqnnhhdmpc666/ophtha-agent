# 简历升级稿

## 推荐项目名

**Ophtha-Agent：面向低资源眼底病问答的证据约束型 Agent 系统**

## 当前可确认版本

**低资源眼底病智能问答系统（RAG + QLoRA + Evidence Traceability）**

构建面向低算力环境的眼底病垂直领域智能问答系统，基于 Qwen2.5-7B-Instruct 与 QLoRA 实现端到端医疗 RAG Pipeline，引入 MarianMT 中英互译、BM25 与向量检索融合、Cross-Encoder 重排序及 Type-Aware 检索策略，并设计 Evidence Traceability 证据溯源与多维自动评测体系，系统比较不同检索策略在问答质量、可信性和安全性上的差异。

### 核心工作

- 基于 Hugging Face 的 EYE-QA-PLUS 眼科问答数据构建领域适配数据，使用 QLoRA/LoRA adapter 对 Qwen2.5-7B-Instruct 进行低资源微调；
- 实现 MarianMT 中英互译、BM25、向量检索和 Cross-Encoder 重排序组成的混合检索 pipeline，完成 vector、hybrid、vector-rerank 和 hybrid-rerank 对照实验；
- 设计结构化 Prompt、知识约束生成和确定性解码策略，降低无证据医学回答风险；
- 按日常建议、疾病定义、决策和紧急情况进行 Type-Aware 检索分析；
- 增加 Evidence Traceability 层，将生成回答与检索证据绑定，并通过 ROUGE-L、BERTScore、关键词覆盖、医学 Checklist 和 LLM-as-a-Judge 进行评测；
- 原有实验中 hybrid-rerank Judge Score 为 4.585，相关指标较基线提升 3–8%；该数字仅代表原项目已完成实验，不代表新 Agent 层已经带来的增益。

## Agent 升级版表述

在原有 RAG 后端之上增加 LangGraph 状态化 Agent 层，封装检索、重排序、证据检查和回答生成工具，实现“问题分类 → 检索 → 重排 → 生成 → 证据验证 → 重检索或安全拒答”的有界工作流；通过 Pydantic 约束工具输入输出，记录可回放的工具轨迹和证据引用，支持本地 Qwen/OpenAI-compatible 模型服务。

### Agent 层可写的新增要点

- 将原有线性 RAG 流程封装为有状态 Agent，保留 FAISS、BM25 和 Cross-Encoder 作为领域检索后端；
- 实现 Evidence Verifier，对回答中的证据 ID 与当前检索结果进行一致性校验；证据不足时最多重检索一次，仍不足则安全拒答；
- 设计 JSONL trace 记录请求、检索候选、验证结果、重试次数和终止状态，支持后续离线评测与训练数据筛选；
- 建立 fixed RAG、Agent-RAG 和后续 QLoRA 版本的统一评测接口，比较证据支持率、引用准确率、回答质量、拒答率、延迟和工具调用成本。

## 暂时不能直接写成结果的内容

以下内容必须等真实 Qwen 模型、真实 FAISS 索引和统一测试集跑完后再填数字：

- Agent 相对 fixed RAG 的提升百分比；
- 证据支持率、引用准确率和安全拒答率；
- Agent 平均工具调用次数与延迟；
- QLoRA 或 ORPO 对 Agent 指标的具体增益；
- 任何“显著降低幻觉”的定量结论。

## 最终简历模板

**Ophtha-Agent：面向低资源眼底病问答的证据约束型 Agent 系统**

技术栈：Python｜LangGraph｜Qwen2.5-7B-Instruct｜QLoRA｜FAISS｜BM25｜Cross-Encoder｜Pydantic｜FastAPI

- 基于混合检索、重排序和 QLoRA 构建低资源眼底病 RAG 系统，并完成多种检索策略及 Type-Aware 路由对照实验；
- 在原有 RAG 后端之上实现 LangGraph 状态化 Agent，统一编排检索、重排序、回答生成和证据验证节点；
- 设计 Evidence Verifier 与有界重检索策略，将回答引用绑定到实际检索文档，证据不足时触发安全拒答；
- 构建可回放 JSONL 工具轨迹和统一评测接口，比较 fixed RAG、Agent-RAG 和 QLoRA 版本在回答质量、证据支持率、引用准确率和推理成本上的差异；
- 原始 hybrid-rerank 实验 Judge Score 达到 4.585，较基线提升 3–8%；Agent 和后训练版本结果待真实实验完成后补充。

