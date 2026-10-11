# 案例 15 · 高并发分布式 Checkpoint 与多实例安全（🔴 高级）

> 一句话：**多个实例共享一个 Postgres checkpoint，同一会话串行写，不同会话并行，状态不打架。**

## 难点是什么
生产是多个实例/进程同时打同一个共享 checkpoint（Postgres）。不加控制会出现：
- 同一会话并发写入相互覆盖，状态损坏/丢消息；
- 每个实例各持一份“假状态”，横向扩展后行为不一致。

## 为什么会发生
默认把 checkpoint 放在内存/单进程里，一旦多实例同时写同一会话，就产生竞态。
共享持久化 + 写入串行化是必经之路。

## 生产级解法（看代码 `main.py`）
1. **共享持久化**：用 Postgres checkpoint，所有实例读同一份状态，
   天然支持横向扩展与断点续跑（见案例 10）。
2. **按 thread_id 串行化**：进程内锁保证“同一场对话”的并发更新不互相覆盖；
   不同 thread_id 完全并行。
3. **多实例说明**：连接同一个 `POSTGRES_URI`；跨进程强一致由 Postgres 行级 upsert +
   事务保证（更严可加分布式锁）。
4. 本机无 Postgres 回退内存 saver，但**内存 saver 不支持多实例**，仅演示。

## 调试清单
- [ ] 同一会话的写入有没有串行化？（测试用 20 线程 + 同会话并发展开验证）
- [ ] 用的是 Postgres 还是内存？内存无法多实例共享。
- [ ] 锁的粒度是 thread_id 吗？（粒度太粗会拖累并发）
- [ ] 分布式锁（跨进程）要不要升级？

## 跑一跑
```bash
# 生产：docker compose up -d postgres，多个实例配相同 POSTGRES_URI
python cases/15_distributed_checkpoint/main.py --message "记住 A=1" --thread-id conv1
python cases/15_distributed_checkpoint/main.py --message "A 是多少？" --thread-id conv1
pytest cases/15_distributed_checkpoint -q    # 20 线程隔离 + 同会话串行用例
```
