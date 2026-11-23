"""Streamlit app: chat with the agent + dashboard.  Run:  streamlit run app.py"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import agent      # noqa: E402
import dashboard  # noqa: E402

st.set_page_config(page_title="Daily Check-in Agent", page_icon="🧠", layout="wide")
page = st.sidebar.radio("View", ["Check-in", "Dashboard"])

if page == "Dashboard":
    dashboard.render()
    st.stop()

if "graph" not in st.session_state:
    st.session_state.graph = agent.build_graph()
if "state" not in st.session_state:
    st.session_state.state = agent.new_state()
state = st.session_state.state

st.title("Daily Check-in")
st.caption("A short chat about your day. Demo prototype, not medical advice.")

for m in state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if state.get("done"):
    st.success("Check-in complete. Open the Dashboard to see it.")

text = st.chat_input("Type your reply... (or /done to finish)", disabled=bool(state.get("done")))
if text:
    work = dict(state)
    work["messages"] = state["messages"] + [{"role": "user", "content": text}]
    try:
        with st.spinner("Thinking..."):
            st.session_state.state = st.session_state.graph.invoke(work)
    except Exception as e:
        st.error(f"The agent hit an error: {e}")
        st.stop()
    st.rerun()

with st.sidebar:
    st.divider()
    if st.button("🔄 New check-in"):
        st.session_state.state = agent.new_state()
        st.rerun()
    with st.expander("Agent internals (for the demo)", expanded=True):
        cov = state.get("covered", [])
        st.write("**Areas covered**")
        for a in agent.AREAS:
            st.write(("✅ " if a in cov else "⬜ ") + a)
        st.write(f"**User turns:** {state.get('turns', 0)} / {agent.MAX_USER_TURNS}")
        st.write(f"**Risk flagged:** {'yes' if state.get('risk') else 'no'}")
        if state.get("scores"):
            st.write("**Extracted scores**")
            st.json(state["scores"])
