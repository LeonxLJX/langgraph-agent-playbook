# 案例 12 · 流式输出与增量呈现（🟡 中级）

> 一句话：**边生成边出字（SSE 推给前端），而不是让用户干等一整段答案。**

## 难点是什么
长答案要几十秒，用户不想干等；前端想要像 ChatGPT 那样逐字显示。
但 Agent 里还夹着工具调用，流式稍不留神就把 token 碎片漏掉或重复。

## 为什么会发生
`graph.invoke()` 是“跑完一次性给全”。要在生成过程中看到增量，得用 `graph.stream` 的
`stream_mode="messages"`，并正确过滤碎片。

## 生产级解法（看代码 `main.py`）
1. agent 节点用 `model.stream()` 产出 token 碎片。
2. 编译图后 `graph.stream(stream_mode="messages")` 拿到**消息级增量**，
   既看到模型打字，也能感知工具调用阶段。
3. FastAPI 用 SSE（sse-starlette）把增量推给前端（`/invoke/stream`）。
4. 关键坑：该模式会同时发出“token 碎片”和“节点返回的完整消息”，
   **用累积去重跳过那条重复的完整消息**，避免前端重复拼接。

## 调试清单
- [ ] 用的是 `stream_mode="messages"` 还是 `"updates"`？要 token 级就用 messages。
- [ ] 有没有去重完整消息？不处理会前端重复。
- [ ] 工具调用阶段要不要展示“正在调工具”？可用 updates 单独感知。
- [ ] SSE 连接超时/中断怎么处理？

## 跑一跑
```bash
python cases/12_streaming/main.py --message "介绍一下扬州"   # 终端实时打印 token
python cases/12_streaming/main.py --serve --port 8000         # /invoke/stream 走 SSE
pytest cases/12_streaming -q
```
