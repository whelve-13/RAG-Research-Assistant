import streamlit as st
import tempfile
import os
from rag_core import process_pdf, build_rag_chain

st.set_page_config(page_title="RAG Academic Assistant", layout="wide")
st.title("📚 RAG Research & Paper Assistant")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF (Research paper, textbook)", type=["pdf"])

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# PDF Processing Trigger
if uploaded_file and groq_api_key:
    if st.session_state.rag_chain is None:
        with st.spinner("Analyzing document and building Chroma vector index..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            retriever = process_pdf(tmp_path)
            st.session_state.rag_chain = build_rag_chain(retriever, groq_api_key)
            os.remove(tmp_path)
            st.success("Document indexed! Start asking questions below.")

# Display historical messages
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# Input handling
user_query = st.chat_input("Ask a question about the uploaded document...")
if user_query:
    if not groq_api_key:
        st.warning("Please provide your Groq API Key in the left sidebar.")
    elif st.session_state.rag_chain is None:
        st.warning("Please upload a PDF document first.")
    else:
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving vector chunks and generating response..."):
                response = st.session_state.rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.write(answer)

                # Render collapsible page citations
                with st.expander("🔍 View Context Sources & Citations"):
                    for i, doc in enumerate(response["context"]):
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Source {i+1} (Page {page}):**")
                        st.caption(doc.page_content[:350] + "...")

                st.session_state.chat_history.append(("assistant", answer))
