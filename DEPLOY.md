# 部署到云端（可远程访问）

每个案例都是一个自包含的 FastAPI 服务，构建方式统一：**根级 Dockerfile + `CASE` 构建参数**。
部署到任意 PaaS 只需两步：选对案例、注入环境变量。

## 0. 快速验证单个服务（本地）

```bash
# 起一个案例（以 01 为例），打开浏览器访问 localhost:8000/docs
docker build -t agent-01 --build-arg CASE=01_tool_call_parse .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx agent-01
```

或用 Docker Compose 一次起全部 15 个：

```bash
docker compose up -d postgres      # 需要持久化的案例依赖的 PG（先起）
docker compose up -d --build       # 起全部，端口 8001~8015
# 或只起一个：docker compose up -d --build case-01
```

## 1. Render（推荐，免费档即可）

**方式 A —— 控制台逐个建：**
1. New → **Web Service** → 连接你的 GitHub 仓库；
2. **Root Directory**：留空（用仓库根）；
3. **Environment**：选 Docker（Dockerfile 已提供）；
4. **Docker Build Context Directory**：`/`（根）；
5. 在 **Build Arguments / Dockerfile** 处注入 `CASE=01_tool_call_parse`（构建时传给 ARG CASE）；
6. 加环境变量 `OPENAI_API_KEY`；
7. 部署完成后，访问 `https://<your-app>.onrender.com/health`，POST `/invoke`。

**方式 B —— Blueprint（一次声明多服务）**：仓库里的 `render.yaml` 是示例，
把它按 15 个案例补全后，在 Render 控制台 **New → Blueprint** 指向本仓库即可。

> Render 免费档有限流（每个服务每月一定时长），学习调试完全够用。

## 2. Fly.io

仓库里的 `fly.toml` 是 case-01 模板。每个案例复制一份改 `app` 与 `CASE`：

```bash
fly launch --no-deploy
# 编辑 fly.toml 的 app 与 build.args.CASE
fly secrets set OPENAI_API_KEY=sk-xxx
fly deploy
```

## 3. 任意 Docker 主机 / 云主机（VPS）

```bash
docker build -t agent --build-arg CASE=01_tool_call_parse .
docker run -d --name agent -p 80:8000 \
  -e OPENAI_API_KEY=sk-xxx \
  agent
```

## 环境变量速查

| 变量 | 必填 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | 是（跑 demo） | LLM key；`LLM_PROVIDER=anthropic` 时用 `ANTHROPIC_API_KEY` |
| `OPENAI_BASE_URL` | 否 | 指向任一 OpenAI 兼容网关/代理 |
| `LLM_MODEL` | 否 | 默认 `gpt-4o-mini` |
| `POSTGRES_URI` | 否 | 案例 10 / 15 的持久化；不填回退内存 |
| `LANGCHAIN_TRACING` / `LANGCHAIN_API_KEY` | 否 | 案例 14 的可观测性 |
| `LOG_LEVEL` | 否 | `INFO` 或 `JSON`（结构化日志） |

## 健康检查

所有服务都暴露 `GET /health`，PaaS 存活探针直接指向它：

```json
{"status": "ok", "service": "case01-tool-call-parse"}
```

## 调用示例

```bash
curl -X POST https://<your-app>/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message":"北京天气怎么样？","thread_id":"u1"}'
# 案例 12 走流式：
curl -N -X POST https://<your-app>/invoke/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"介绍一下扬州"}'
```
