"""Chat UI. Run with: uv run streamlit run app/streamlit_app.py"""

import json

import streamlit as st

from raposa.personas import PERSONAS
from raposa.runtime import build_agent

st.set_page_config(page_title="Raposa Agent", page_icon="🦊")
st.title("🦊 Raposa Agent")
st.caption("Ask anything about Cruzeiro: last results, next games, head-to-head, club history.")

with st.sidebar:
    persona = st.radio(
        "Persona",
        options=list(PERSONAS),
        format_func=lambda key: PERSONAS[key].label,
        help="Same facts, different voice.",
    )
    st.caption(PERSONAS[persona].description)
    show_tools = st.toggle("Show tool calls", value=False)
    if st.button("New conversation"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Como foi o último jogo do Cruzeiro?"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Consultando a Toca da Raposa..."):
        reply = build_agent().ask(question, persona=persona, history=history)
        st.markdown(reply.text)
        if show_tools:
            for call in reply.tool_calls:
                with st.expander(f"🔧 {call.name}({json.dumps(call.input, ensure_ascii=False)})"):
                    st.code(call.output, language="json")
    st.session_state.messages.append({"role": "assistant", "content": reply.text})
