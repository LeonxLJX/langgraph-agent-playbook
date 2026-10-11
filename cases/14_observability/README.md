# 案例 14 · 生产可观测性与成本（🔴 高级）

> 一句话：**别让 Agent 上生产后成黑盒 —— 上链追踪 + 节点插桩 + 运行级指标，成本延迟都可监控。**

## 难点是什么
Agent 上生产后看不见：不知道每轮跑了哪些节点、花了多少 token、哪一步最慢、哪里报错。
没有可观测性，线上问题只能靠猜；token 成本也无法预算。

## 为什么会发生
很多人把图调通就上线，没接追踪、没记指标。Agent 是多步不确定运行，
比普通服务更需要可观测性。

## 生产级解法（看代码 `main.py`）
1. **LangSmith 追踪**：`LANGCHAIN_TRACING=true` 后整条图自动上链，
   可按 thread_id 回看每步输入输出与耗时（`core/tracing.py` 统一开关）。
2. **节点级插桩**：装饰器给每个节点计时 + 记录 token 估算 + 捕获异常，落结构化日志。
3. **运行级指标**：每次 invoke 汇总 steps、节点调用数、估算 token、耗时，
   随结果返回，可喂给监控大盘做成本/延迟告警。

## 调试清单
- [ ] 追踪是否“显式开启”？（避免把调试流量误打进计费）
- [ ] 每个节点有计时/异常捕获吗？
- [ ] token 估算有吗？（没有成本控制就是盲盒）
- [ ] 指标有没有落到结构化日志（可被 CloudWatch/ELK 收集）？

## 跑一跑
```bash
# 可选：开启 LangSmith
#   $env:LANGCHAIN_TRACING="true"; $env:LANGCHAIN_API_KEY="lsv2_xxx"
python cases/14_observability/main.py --message "你好"    # 看 meta 里的观测指标
pytest cases/14_observability -q
```
