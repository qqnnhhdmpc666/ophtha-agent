# Training and alignment reservation

这一层只预留训练接口，不在本地 Mock 阶段启动训练。

## 固定分工

```text
Base model
  ↓
QLoRA / SFT       学会领域回答格式、证据引用和工具使用
  ↓ (optional)
ORPO              在相同问题上偏向合规回答，抑制无证据回答
  ↓ (future)
GRPO              需要真实环境、在线 rollout 和 verifier reward
```

SFT 的正例必须来自通过证据检查的回答；ORPO 的 `chosen/rejected` 必须具有相同 prompt，且 rejected 的失败原因可审计；GRPO 不接受离线 JSON 伪装成在线轨迹。

## 模型角色

`model_registry.example.yaml` 预留了三种角色：

- `policy_model`：实际 Agent 模型，例如本地 Qwen3-4B；
- `teacher_or_evaluator`：可选的较大模型，只用于生成/评估，不与 policy 结果混称；
- `embedding_and_reranker`：检索组件，不参与语言模型训练。

模型路径、checkpoint 和 API key 不进入 Git。

## 阶段开关

- `sft_qlora.yaml`：可运行配置模板，但必须先通过数据契约和 baseline gate；
- `orpo_qlora.yaml`：可选离线偏好对齐，默认关闭；
- `grpo_reserved.yaml`：仅记录接口和前置条件，`enabled: false`，不会误启动训练。

