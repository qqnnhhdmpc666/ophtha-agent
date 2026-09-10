# 简历最终替换稿

> 以原简历为底稿：保留原项目成果，只将 Evidence Traceability 一条扩展为 Agent 设计。

## 项目名称

**低资源眼底病智能问答系统（RAG + QLoRA）**

## 项目描述

构建面向低算力环境的眼底病垂直领域智能问答系统，基于 Qwen2.5-7B-Instruct + QLoRA 实现端到端医疗 RAG Pipeline，引入混合检索、Cross-Encoder 重排序与 Type-Aware 策略分析；在原有系统上增加轻量级状态化 Agent 外壳，统一编排检索、生成、证据验证和安全终止。

## 技术栈

Python｜QLoRA｜FAISS｜BM25｜Cross-Encoder｜Sentence-Transformers｜LangGraph｜Pydantic｜FastAPI

## 核心工作

- 基于 Hugging Face EYE-QA-PLUS 眼科问答数据完成 QLoRA 领域适配，构建面向低资源环境的医疗问答基线；
- 实现 MarianMT 中英互译、BM25 与向量检索加权融合，并结合 Cross-Encoder 重排序，完成 vector、hybrid、vector-rerank 和 hybrid-rerank 对照实验；
- 设计 Type-Aware 检索策略，针对日常建议、疾病定义、决策和紧急情况分析不同检索配置的适用性；
- 设计结构化 Prompt、知识约束生成和确定性解码策略，减少无证据回答和不安全医疗建议；
- 在原线性 RAG 上封装 LangGraph Agent 工作流，将检索、重排、生成与证据核验拆分为可组合节点，通过状态转移统一控制重检索、拒答与轨迹记录；
- 原有 hybrid-rerank 实验 Judge Score 达到 4.585，相关指标较基线提升 3–8%；Agent overlay 的新增收益在统一真实测试完成后填入，不提前虚构。

## 一句话版本

基于 QLoRA 和混合检索构建低资源眼底病 RAG 系统，并设计 LangGraph 状态化 Agent 外壳，通过证据验证、有界重检索和可回放轨迹提升系统的可控性与工程完整性。
