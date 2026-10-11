# 案例 01 · 工具调用参数解析崩溃（🔵 初级）

> 一句话：**模型返回的 tool_call 参数不对，Agent 就整个崩掉 —— 你要把它变成“能自己纠正”的消息，而不是让它中断。**

## 难点是什么

模型说“我要调 `get_weather`”，但参数可能是：

- **非法 JSON**：`{"city": 北京}`（字符串没加引号）
- **缺必填字段**：`{}` 或 `{"province": "北京"}`（字段名写错）
- **类型不对**：`{"city": 123}`（传了数字）

直接用 langgraph 自带的 `ToolNode` 执行，以上任何一种都会抛 `ValidationError`，**整次 Agent 运行直接中断**，用户拿到的是报错而不是回答。

## 为什么会发生

LLM 是概率模型，`bind_tools` 只是“软约束”。即使工具 schema 写得很严格，模型照样可能产出不合规的参数。生产环境是海量并发，**不能赌模型每次都对**，必须让“出错”这件事本身是可控的。

## 生产级解法（看代码 `main.py`）

1. **不用裸 `ToolNode`，用「守卫工具节点」`_guarded_tools_node`**：
   - 工具名不存在 → 回传 `[工具不存在]`，并把可用工具列表告诉模型，让它重新选；
   - 参数校验/执行失败 → 用 `try/except` 包住，回传 `[工具调用失败] + 错误原因 + 原参数`；
   - 关键点：**把失败变成 `ToolMessage` 回传，而不是抛异常** —— 模型读到错误会自己修正参数再调一次（自愈）。
2. **兜底 `recursion_limit`**：即使模型反复错，图也不会无限跑下去。

## 调试清单（出问题时按顺序查）

- [ ] 是否所有 tool_call 都被 try/except 包住？裸 `ToolNode` 没有，会崩。
- [ ] 错误信息里有没有给模型“下一步该干嘛”的提示（可用工具/该修哪个参数）？
- [ ] `tool_call_id` 是否和 `AIMessage.tool_calls` 里的 id 对上？（对不上模型没法把结果和调用关联）
- [ ] 是否设置了 `recursion_limit` 防死循环？

## 跑一跑

```bash
# 一次性演示（把城市换成别的，看看它怎么调工具）
python cases/01_tool_call_parse/main.py --message "北京天气是多少，转成华氏？"

# 起成 HTTP 服务（云部署的本地预览）
python cases/01_tool_call_parse/main.py --serve --port 8000
# curl -X POST localhost:8000/invoke -H 'Content-Type: application/json' -d '{"message":"北京天气？"}'

# 离线测试（不花 token，验证守卫逻辑）
pytest cases/01_tool_call_parse -q
```

**想亲自感受“崩溃”和“自愈”的区别**：把 `main.py` 里 `tools` 节点换成 langgraph 的 `ToolNode(TOOLS)`，再跑 `pytest`，会看到测试直接红 —— 这就是你上生产前要消灭的崩溃点。
