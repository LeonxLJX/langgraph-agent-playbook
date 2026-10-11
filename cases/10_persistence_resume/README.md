# 案例 10 · 持久化与断点续跑（🟡 中级）

> 一句话：**把状态落库（Postgres），进程重启 / 换实例都记得，还能断点续跑。**

## 难点是什么
多轮对话 / 长任务，一旦进程重启或并发请求，状态就没了；不持久化，一切“记住的东西”
都随进程消失，也做不了断点续跑和时间旅行。

## 为什么会发生
默认的 `InMemorySaver` 只存在内存里，进程一结束就清空。生产要横向扩展、要抗重启，
必须把 checkpoint 落到共享存储。

## 生产级解法（看代码 `main.py`）
1. 编译图时传 checkpointer（内存或 Postgres），所有消息/状态按 `thread_id` 落库。
2. 同一 `thread_id` 再次调用自动带历史 —— **换一个全新图实例 + 同一个库，记忆也在**。
3. 生产用 Postgres（`core/checkpoint.py` 统一入口）；本机无 PG 时回退内存做演示。
4. 与案例 08 的 `interrupt` 天然配合：审批断点也靠 checkpoint 存续。

## 调试清单
- [ ] 每次调用都传了 `thread_id` 吗？没有它没法区分会话。
- [ ] 用的是 Postgres 还是内存？生产必须 Postgres（内存不支持多实例/重启）。
- [ ] Postgres 建表（`setup()`）做了吗？首次会失败。
- [ ] 跨实例测试过吗？（本案例测试用两个图实例 + 同一 saver 验证了这一点）

## 跑一跑
```bash
# 内存演示
python cases/10_persistence_resume/main.py --message "记住我叫小王" --thread-id u1
python cases/10_persistence_resume/main.py --message "我叫什么？"   --thread-id u1   # 记得
python cases/10_persistence_resume/main.py --message "我叫什么？"   --thread-id u2   # 不记得
# 生产：docker compose up -d postgres，配 POSTGRES_URI 后重启服务也保留
pytest cases/10_persistence_resume -q
```
