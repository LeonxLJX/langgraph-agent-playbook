# 案例 02 · 上下文超长与 token 溢出（🔵 初级）

> 一句话：**多轮对话/反复工具调用让 messages 无限长，最后超模型窗口直接崩 —— 你要在超限前把历史“压缩”掉。**

## 难点是什么

Agent 每走一轮，`messages` 就多一条。几十轮下来：

- 超出模型 context window → 调用报错，Agent 中断；
- 即便不报错，token 按量计费，长上下文烧钱很快；
- 响应变慢（长 prompt 处理慢）。

## 为什么会发生

很多新手把整个对话历史原封不动塞给模型。聊天一长，没有人“修剪”。生产里必须主动做**上下文预算管理**。

## 生产级解法（看代码 `main.py`）

1. **自定义 reducer `_messages_reducer`**：既能用 langgraph 的 `add_messages` 正常追加消息，又支持用一个 `__reset__` 标记**整体替换历史**（压缩的核心机制）。
2. **`estimate_tokens` 估算上下文大小**（启发式，零额外依赖；生产可换 tiktoken 精确计数）。
3. **超过阈值 → `summarize` 节点**：把“旧消息”交给模型总结成一条 `SystemMessage` 摘要，替换掉原文，保留最近 `keep_recent` 条。
4. **压缩后基于摘要再作答一次**，让对话“无缝延续”，而不是压缩完就丢答案。

## 调试清单

- [ ] 有没有给 messages 配 reducer？默认 `add_messages` 只能追加，**没法删历史**，压缩会失效。
- [ ] 阈值设对了吗？设太紧会频繁压缩、丢细节；太松等于不压缩。
- [ ] 压缩后历史总长是否稳定低于阈值？（否则会反复压缩 → 死循环，配合案例 03 的兜底）
- [ ] 摘要里有没有保留“用户真正诉求 + 已确认的关键事实”？丢关键信息模型会失忆。

## 跑一跑

```bash
# 一次性演示（长输入触发压缩，观察 meta.compressed）
python cases/02_context_truncate/main.py --message "$('天气'*200)"

python cases/02_context_truncate/main.py --serve --port 8000

# 离线测试
pytest cases/02_context_truncate -q
```
