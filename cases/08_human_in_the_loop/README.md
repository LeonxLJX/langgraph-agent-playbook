# 案例 08 · 人类介入审批（Human-in-the-Loop）（🟡 中级）

> 一句话：**Agent 想干高风险动作（转账/删数据/发邮件），先暂停等真人拍板，再决定执行或拒绝。**

## 难点是什么
不可逆 / 高成本动作（转账、删数据、下单、发消息），Agent 直接执行代价巨大。
生产里必须**暂停下来，把提案交给真人，等确认后再继续**。

## 为什么会发生
很多 Agent 把“能调工具”当“应该自动调”。但工具是有副作用的，尤其不可逆操作，
必须有人类把关的闸门。

## 生产级解法（看代码 `main.py`）
1. `SENSITIVE_TOOLS` 白名单：命中敏感工具的调用一律进审批节点。
2. `dispatch` 节点调用 `interrupt({proposal})`：图在此暂停，提案交到外部（API/UI）。
3. 外部用 `Command(resume={"approve": True/False})` 续跑：
   - True → 真正执行工具；
   - False → 回传 `[未批准]`，工具**不执行**（零副作用）。
4. 需要 checkpointer（内存或 Postgres）才能断点续跑 —— 生产用 Postgres（见案例 10）。

## 调试清单
- [ ] 敏感工具清单设全了吗？（转账/删除/发送 都要覆盖）
- [ ] 审批前工具真的没执行吗？（测试里用计数器验证了这一点）
- [ ] 用了 checkpointer 吗？没有它 `interrupt` 无法断点续跑。
- [ ] 审批结果是阻塞外部等待，还是可以异步续跑？（本案例支持 `approver` 异步续跑）

## 跑一跑
```bash
python cases/08_human_in_the_loop/main.py --message "转账 100 元给小王"   # 会先暂停要审批
pytest cases/08_human_in_the_loop -q    # 含“通过执行”和“拒绝不执行”两个用例
```
