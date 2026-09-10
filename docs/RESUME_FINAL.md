# 简历最终替换稿

> 以原简历为底稿，保留原项目成果，并将 Agent 设计自然融入项目描述。

## 项目名称

**低资源眼底病智能问答系统（RAG + QLoRA）**

## 项目描述

构建面向低算力环境的眼底病垂直领域智能问答 Agent，基于 Qwen2.5-7B-Instruct + QLoRA 实现领域适配，以 MarianMT 中英互译、BM25、向量检索和 Cross-Encoder 重排序构成混合 RAG Pipeline，并通过 LangGraph 状态化工作流统一编排问题分类、检索、重排、生成、证据验证和安全终止。

## 技术栈

Python｜QLoRA｜FAISS｜BM25｜Cross-Encoder｜Sentence-Transformers｜LangGraph｜Pydantic｜FastAPI

## 核心工作

- 基于 Hugging Face EYE-QA-PLUS 眼科问答数据完成 QLoRA 领域适配，构建面向低资源环境的医疗问答基线；
- 实现 MarianMT 中英互译、BM25 与向量检索加权融合，并结合 Cross-Encoder 重排序，完成 vector、hybrid、vector-rerank 和 hybrid-rerank 对照实验；
- 设计 Type-Aware 检索策略，针对日常建议、疾病定义、决策和紧急情况分析不同检索配置的适用性；
- 设计结构化 Prompt、知识约束生成和确定性解码策略，减少无证据回答和不安全医疗建议；
- 将 RAG Pipeline 设计为基于 LangGraph 的状态化 Agent 工作流，把检索、重排、生成和证据核验组织为可组合节点，并通过状态转移控制重检索、拒答与轨迹记录；
- hybrid-rerank 配置 Judge Score 达到 4.585，相关指标较基线提升 3–8%；

## 一句话版本

基于 QLoRA、混合检索与 LangGraph 构建低资源眼底病问答 Agent，通过证据验证、有界重检索和可回放轨迹提升回答的可靠性与系统可控性。
