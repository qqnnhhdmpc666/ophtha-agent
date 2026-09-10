# Training data contract

## SFT

每条记录至少包含：

```json
{"id":"...", "messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}], "evidence_ids":["..."]}
```

只保留通过 verifier 的 assistant 目标段。用户消息、检索文档和工具 observation 不作为语言模型监督目标。

## ORPO

```json
{"id":"...", "prompt":[...], "chosen":[...], "rejected":[...], "failure_type":"unsupported_answer"}
```

`chosen` 和 `rejected` 必须共享同一个 prompt；失败原因、验证结果和来源轨迹必须能回溯。

## GRPO

```json
{"task_id":"...", "initial_state":{}, "tool_schema":{}, "prompt":"...", "verifier":"..."}
```

这里只保存任务注册表，不把预先写好的答案当作在线 rollout。真实 rollout、状态变化和 reward 必须由运行时产生。

