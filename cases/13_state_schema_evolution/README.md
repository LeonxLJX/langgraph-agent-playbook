# 案例 13 · 复杂状态机图编排与 Schema 演进（🔴 高级）

> 一句话：**图一复杂就乱 —— 拓扑要集中声明、状态 schema 要带版本、加字段要向后兼容。**

## 难点是什么
生产图常有多个分支 + 条件路由 + 回流。一复杂就容易：
- 条件路由写错分支名 / 指向不存在的节点（运行期才炸）；
- 加了新字段没做兼容，老 checkpoint 反序列化直接 KeyError；
- 状态字段随手改，升级后线上历史状态全崩。

## 为什么会发生
把图写成一堆散落的 `add_edge` / `add_conditional_edges`，没有集中管理拓扑；
状态 schema 没有版本意识，随加随改。

## 生产级解法（看代码 `main.py`）
1. **拓扑集中声明**：一张小表说清每个分支去向（`TOPOLOGY`），杜绝“箭头满天飞”难排查。
2. **schema_version 字段**：新增字段一律“可选 + `.get(默认值)`”向后兼容。
3. **聚合用 reducer**：`Annotated[list, add]` 让多分支产出可合并。
4. **条件路由返回明确分支名**，与 `add_conditional_edges` 的分支表对死；
   未知分支安全回退（本案例回退 direct）。

## 调试清单
- [ ] 分支名和节点名对得上吗？（写错会运行期报错）
- [ ] 新增字段有没有默认值/可选？老 checkpoint 反序列化会 KeyError。
- [ ] 分支表里有没有 END？防“无路可走”。
- [ ] 状态 schema 变更有没有记录版本号？

## 跑一跑
```bash
python cases/13_state_schema_evolution/main.py --message "写一篇新品文案"   # writer 分支
python cases/13_state_schema_evolution/main.py --message "查竞品数据"       # research 分支
pytest cases/13_state_schema_evolution -q
```
