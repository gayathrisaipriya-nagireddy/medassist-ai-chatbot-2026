import streamlit as st
from chatbot import answer

st.set_page_config(page_title="MedAssist AI", page_icon="🩺")
st.title("🩺 MedAssist AI")
st.caption("RAG-based general medical information assistant")
st.warning("Educational information only. This tool cannot diagnose or prescribe. For urgent symptoms, seek immediate medical care.")
if "messages" not in st.session_state:
    st.session_state.messages = []
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)
question = st.chat_input("Ask a general health-information question")
if question:
    st.session_state.messages.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching trusted-source summaries..."):
            response = answer(question)
        st.markdown(response)
    st.session_state.messages.append(("assistant", response))
