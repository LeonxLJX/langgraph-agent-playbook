# 根 Dockerfile：所有案例共用一个镜像，用构建参数 CASE 切换。
# 构建示例（以案例 01 为例）：
#   docker build -t agent-01 --build-arg CASE=01_tool_call_parse .
#   docker run -p 8000:8000 -e OPENAI_API_KEY=... agent-01
ARG PYTHON=3.13-slim
FROM python:${PYTHON}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝整个仓库
COPY . .

# 切换到目标案例
ARG CASE
ENV CASE_DIR=/app/cases/${CASE}
WORKDIR ${CASE_DIR}

EXPOSE 8000

# 默认以 HTTP 服务方式启动（可被 Render / Fly.io / Compose 拉起）
CMD ["python", "main.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]
