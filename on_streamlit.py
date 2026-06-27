import streamlit as st
import sys

from on_terminal import state, ingest, build_context, llm, embed, np
from on_terminal import BUFFER_LIMIT, SUMMARY_TRIGGER, VECTOR_TOP_K

st.set_page_config(page_title="Hierarchical Memory Chatbot", layout="wide")
st.title("Hierarchical Memory Chatbot")

# ── Init session state ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "state_snapshots" not in st.session_state:
    st.session_state.state_snapshots = []

def snapshot():
    s = state
    q_emb = embed(st.session_state.messages[-1]["content"]) if st.session_state.messages else None
    hits = []
    if q_emb is not None:
        hits = sorted(
            [v for v in s["vectors"] if v["turn"] != s["turn"]],
            key=lambda v: float(np.dot(q_emb, v["emb"])), reverse=True
        )[:VECTOR_TOP_K]
    st.session_state.state_snapshots.append({
        "turn": s["turn"],
        "buffer_len": len(s["buffer"]),
        "summary": s["summary"],
        "entities": dict(s["entities"]),
        "top_vector_hits": [(h["role"], h["content"][:80]) for h in hits],
    })

# ── Chat input ─────────────────────────────────────────────────
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    ingest("user", prompt)
    system, buf = build_context(prompt)
    reply = llm(system, buf)
    ingest("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    snapshot()

# ── Display chat ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Sidebar: Memory layers ─────────────────────────────────────
if st.session_state.state_snapshots:
    snap = st.session_state.state_snapshots[-1]

    with st.sidebar:
        st.subheader(f"Turn {snap['turn']}")

        with st.expander(f"Layer 1 — Buffer ({snap['buffer_len']}/{BUFFER_LIMIT})", True):
            st.metric("Messages in buffer", snap["buffer_len"])

        with st.expander(f"Layer 2 — Summary (trigger at {SUMMARY_TRIGGER})", True):
            st.text(snap["summary"] if snap["summary"] else "(no summary yet)")

        with st.expander(f"Layer 3 — Vector top-{VECTOR_TOP_K}", True):
            if snap["top_vector_hits"]:
                for role, text in snap["top_vector_hits"]:
                    st.caption(f"{role.upper()}")
                    st.text(text)
            else:
                st.text("(no relevant hits)")

        with st.expander(f"Layer 4 — Entities ({len(snap['entities'])})", True):
            if snap["entities"]:
                for name, info in snap["entities"].items():
                    st.markdown(f"- **{name}** ({info['type']}, {info['count']}x)")
            else:
                st.text("(no entities tracked)")

        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages.clear()
            st.session_state.state_snapshots.clear()
            state["buffer"] = []
            state["summary"] = ""
            state["vectors"] = []
            state["entities"] = {}
            state["turn"] = 0
            state["history"] = []
            st.rerun()
else:
    with st.sidebar:
        st.info("Send a message to start.")
