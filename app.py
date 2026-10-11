"""LangGraph Agent 排障案例 · 网页调试台（Streamlit）。

在网页里逐个试 01~15 案例：左侧选案例，中间对话，右侧/折叠区看每次运行的元信息。
- 未配置 OPENAI_API_KEY 时走离线假模型（也能点通流程），配置后在云端 Settings→Secrets
  添加 OPENAI_API_KEY 即切真实 LLM。
- 案例 08 人类审批：当返回 needs_approval 时出现"同意 / 拒绝"按钮。
"""
import importlib.util
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import get_settings

# ------------------------------------------------------------------ 案例目录
CASES = [
    ("01_tool_call_parse", "初级", "01 · 工具调用参数解析崩溃"),
    ("02_context_truncate", "初级", "02 · 上下文超长与 token 溢出"),
    ("03_loop_guard", "中级", "03 · 死循环熔断"),
    ("04_hallucinated_tool", "中级", "04 · 幻觉工具名拦截"),
    ("05_retry_backoff", "中级", "05 · 上游限流退避重试"),
    ("06_self_healing", "中级", "06 · 节点异常自愈"),
    ("07_multi_agent_routing", "中级", "07 · 多智能体路由与上下文隔离"),
    ("08_human_in_the_loop", "中级", "08 · 人类介入审批"),
    ("09_parallel_tools_idempotent", "中级", "09 · 并行工具调用与幂等"),
    ("10_persistence_resume", "中级", "10 · 持久化与断点续跑"),
    ("11_structured_output", "中级", "11 · 结构化输出校验重试"),
    ("12_streaming", "中级", "12 · 流式输出与增量呈现"),
    ("13_state_schema_evolution", "高级", "13 · 复杂状态机与 Schema 演进"),
    ("14_observability", "高级", "14 · 生产可观测性与成本"),
    ("15_distributed_checkpoint", "高级", "15 · 高并发分布式 Checkpoint"),
]

SETTINGS = get_settings()


def load_case(case_dir: str):
    """按文件路径加载案例模块（避免 package 约束）。"""
    key = f"_mod_{case_dir}"
    if key in st.session_state:
        return st.session_state[key]
    path = os.path.join(ROOT, "cases", case_dir, "main.py")
    spec = importlib.util.spec_from_file_location(f"case_{case_dir}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    st.session_state[key] = mod
    return mod


def get_runner(case_dir: str):
    """懒加载并缓存 runner；08 特殊返回 (approval, runner, approver)。"""
    key = f"_runner_{case_dir}"
    if key not in st.session_state:
        mod = load_case(case_dir)
        if case_dir == "08_human_in_the_loop":
            runner, approver = mod.build_approval_flow(SETTINGS)
            st.session_state[key] = ("approval", runner, approver)
        else:
            runner = mod.build_runner(SETTINGS)
            st.session_state[key] = ("plain", runner)
    return st.session_state[key]


def thread_id_for(case_dir: str) -> str:
    return f"streamlit-{case_dir}"


def chat_key(case_dir: str) -> str:
    return f"_chat_{case_dir}"


# ------------------------------------------------------------------ 页面
st.set_page_config(page_title="LangGraph Agent 案例调试台", page_icon="🤖", layout="wide")
st.title("🤖 LangGraph Agent 排障案例 · 网页调试台")
st.caption("初级 2 · 中级 10 · 高级 3 —— 左侧选案例，逐个远程调试。未配 key 时走离线假模型。")

sel = st.sidebar.radio("选择案例", CASES, format_func=lambda x: f"{x[1]}｜{x[2]}")
case_dir, difficulty, label = sel

with st.sidebar.expander("怎么配真实 LLM"):
    st.markdown(
        "默认走离线假模型（演示流程、不花 token）。要真实 LLM：在 "
        "Streamlit Cloud 的 **Settings → Secrets** 添加 `OPENAI_API_KEY` 即可。"
    )

st.subheader(label)
st.sidebar.markdown(f"**难度：{difficulty}**")

# 每个案例独立的会话消息
msgs = st.session_state.setdefault(chat_key(case_dir), [])

for role, content in msgs:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("输入要测的内容（如：北京天气并转成华氏）")

if prompt:
    msgs.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            kind, runner = get_runner(case_dir)[0], get_runner(case_dir)[1]
            answer, meta = runner(prompt, thread_id_for(case_dir))
            placeholder.markdown(answer or "(等待输入 / 需要审批)")
            with st.expander("本次运行元信息 (meta)"):
                st.json(meta)

            if kind == "approval" and meta.get("needs_approval"):
                _, _, approver = get_runner(case_dir)
                st.warning("🛡️ 该操作需要人工审批（案例 08 的中断点）")
                c1, c2 = st.columns(2)
                if c1.button("✅ 同意", key=f"ok_{case_dir}"):
                    ans2, m2 = approver(thread_id_for(case_dir), True)
                    msgs.append(("assistant", ans2 or "(已处理)"))
                    st.rerun()
                if c2.button("🚫 拒绝", key=f"no_{case_dir}"):
                    ans2, m2 = approver(thread_id_for(case_dir), False)
                    msgs.append(("assistant", ans2 or "(已处理)"))
                    st.rerun()
            else:
                msgs.append(("assistant", answer))
        except Exception as exc:  # noqa: BLE001 - 把异常显示给用户而非崩溃
            placeholder.error(f"运行出错：{exc!r}")
            st.session_state.pop(f"_runner_{case_dir}", None)
