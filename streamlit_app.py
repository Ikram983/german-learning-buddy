# streamlit_app.py
from __future__ import annotations

import sys
from pathlib import Path

# Same path fix as your CLI app
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from german_buddy.graph import build_graph, run_turn


# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="German Learning Buddy",
    page_icon="🇩🇪",
    layout="centered",
)
st.title("🇩🇪 German Learning Buddy")
st.caption("Type a German sentence and get instant feedback.")


# ============================================================
# Cache the graph (load once, reuse across messages)
# ============================================================
@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()


# ============================================================
# Session state: chat history
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content}]

if "history" not in st.session_state:
    st.session_state.history = []  # your run_turn history


# ============================================================
# Display existing messages
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============================================================
# Chat input
# ============================================================
if prompt := st.chat_input("Type a German sentence..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("Checking..."):
            try:
                answer, new_history = run_turn(
                    graph, prompt, st.session_state.history
                )
                st.session_state.history = new_history
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# ============================================================
# Sidebar: clear chat
# ============================================================
with st.sidebar:
    st.markdown("## 🇩🇪 German Learning Buddy")
    st.markdown(
        "Paste a German sentence. I'll check:\n"
        "- ✅ Word order\n"
        "- ✅ Spelling\n"
        "- ✅ Capitalization\n"
        "- ✅ Grammar\n"
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.history = []