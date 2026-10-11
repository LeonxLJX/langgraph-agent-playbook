# langgraph-agent-playbook

**生产级 LangGraph Agent 排障案例集** —— 15 个真实工作中会踩的 Agent 工程难点，每个都是一份
**可直接运行的完整生产代码 + 调试教学 + 云部署方案**。

> 难度分布：🔵 初级 2 · 🟡 中级 10 · 🔴 高级 3

## 为什么要有这个仓库

搭 Agent demo 很容易，上生产全是坑。这个仓库把 Agent 工程里**最高频、最容易翻车**的 15 个难点，
每个都写成一份能跑起来的完整案例，配 **离线可复现的测试**（不花 token 也能验证逻辑），
并且每个案例都能用 Docker 一键部署到 **Render / Fly.io / 任意 PaaS**，供你逐个调、逐个学。

## 案例总览

| # | 难度 | 案例 | 你解决的是哪个难点 |
|---|------|------|-------------------|
| 01 | 🔵 初级 | 工具调用参数解析崩溃 | LLM 输出非法 JSON / schema 不匹配导致工具调用炸 |
| 02 | 🔵 初级 | 上下文超长与 token 溢出 | 多轮对话 context 爆掉，怎么压缩 |
| 03 | 🟡 中级 | 死循环熔断 | Agent 卡死循环，最大步数兜底 |
| 04 | 🟡 中级 | 幻觉工具名拦截 | 模型调用不存在的工具，非法调用怎么拦 |
| 05 | 🟡 中级 | 上游限流退避重试 | 429 / 超时，指数退避不雪崩 |
| 06 | 🟡 中级 | 节点异常自愈 | 单点失败后状态一致性补偿 |
| 07 | 🟡 中级 | 多智能体路由与上下文隔离 | Supervisor 编排，上下文不串 |
| 08 | 🟡 中级 | 人类介入审批 | 高危工具动作先人工确认再执行 |
| 09 | 🟡 中级 | 并行工具调用与幂等 | 并发执行、结果合并、防重复副作用 |
| 10 | 🟡 中级 | 持久化与断点续跑 | Postgres checkpoint，重启不丢、可续跑 |
| 11 | 🟡 中级 | 结构化输出校验重试 | 输出不符合 schema，带反馈重试 |
| 12 | 🟡 中级 | 流式输出与增量呈现 | streaming，边生成边出 |
| 13 | 🔴 高级 | 复杂状态机与 schema 演进 | 图编排、条件路由、State 迁移的坑 |
| 14 | 🔴 高级 | 生产可观测性与成本 | LangSmith 追踪、指标、成本控制 |
| 15 | 🔴 高级 | 高并发分布式 checkpoint | 多实例并发、分布式持久化安全 |

## 目录结构

```
langgraph-agent-playbook/
├── core/                  # 所有案例共享的生产级基础设施
│   ├── settings.py        # 集中配置（pydantic-settings）
│   ├── llm.py             # LLM 工厂（超时/重试）
│   ├── checkpoint.py      # 持久化工厂（内存/Postgres）
│   ├── tracing.py         # LangSmith 开关
│   ├── server.py          # FastAPI 服务脚手架
│   ├── cli.py             # 统一 CLI / 服务启动入口
│   └── fake_llm.py        # 离线测试用可控假 LLM
├── cases/
│   ├── 01_tool_call_parse/
│   │   ├── main.py        # 生产代码：LangGraph 应用 + CLI + FastAPI
│   │   ├── test_01_....py # 离线测试（假 LLM，不花 token）
│   │   └── README.md      # 难点讲解 + 调试步骤
│   └── ...                # 02~15 同理
├── Dockerfile             # 根级镜像，用构建参数 CASE 切换案例
├── docker-compose.yml     # 一键本地起全部/单个
├── render.yaml            # Render 云部署蓝图（示例）
├── fly.toml               # Fly.io 部署模板（示例）
├── DEPLOY.md              # Render / Fly.io / Docker 部署教程
└── .github/workflows/ci.yml  # 每次提交自动跑全部测试
```

## 快速开始

```bash
# 1. 环境已统一装在你的 Anaconda（D:\ProgramData\anaconda3）里。
#    新机器可按需装：python -m venv .venv && pip install -r requirements-dev.txt

# 2. 配好 LLM key（可选，跑测试不需要）
copy .env.example .env   # 填入 OPENAI_API_KEY 或 ANTHROPIC_API_KEY

# 3. 离线跑通全部案例测试（不花 token）
pytest cases -q

# 4. 单案例跑 demo（交互式演示该难点的解法）
python cases/01_tool_call_parse/main.py --message "查一下北京天气并转成摄氏度"

# 5. 本地起成 HTTP 服务（云部署的本地预览）
python cases/01_tool_call_parse/main.py --serve --port 8000
# curl -X POST localhost:8000/invoke -H 'Content-Type: application/json' -d '{"message":"你好"}'
```

云部署（Render / Fly.io / Docker Compose）的完整步骤见 **[DEPLOY.md](DEPLOY.md)**。

## 怎么读这个仓库

每个案例的 `README.md` 都按同一模板写，方便你对照学习：

1. **难点是什么** —— 一句话说清这个坑
2. **为什么会发生** —— 根因拆解
3. **生产级解法** —— 代码里怎么解决的，关键点标了注释
4. **调试清单** —— 出问题时按什么顺序排查
5. **跑一跑** —— CLI demo + 测试命令

## 许可

MIT —— 随意取用、修改、商用。
