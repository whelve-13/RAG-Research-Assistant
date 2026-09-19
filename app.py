import streamlit as st
import tempfile
import os
from rag_core import process_pdf, build_rag_chain

st.set_page_config(page_title="RAG Academic Assistant", layout="wide")

st.title("📚 RAG Research & Paper Assistant")

st.sidebar.header("Configuration")

# Retrieve key from Streamlit Cloud Secrets or ask user locally
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

uploaded_file = st.sidebar.file_uploader("Upload a PDF (Research paper, textbook)", type=["pdf"])

# Initialize session state for the retrieval chain and chat history
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Process uploaded document
if uploaded_file and groq_api_key:
    if st.session_state.rag_chain is None:
        with st.spinner("Processing document embeddings..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                retriever = process_pdf(tmp_path)
                st.session_state.rag_chain = build_rag_chain(retriever, groq_api_key)
                st.sidebar.success("Document indexed successfully!")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User query input
if user_query := st.chat_input("Ask a question about the uploaded document..."):
    if not groq_api_key:
        st.warning("Please provide a Groq API Key to proceed.")
    elif not uploaded_file:
        st.warning("Please upload a PDF document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                raw_response = st.session_state.rag_chain.invoke({"input": user_query})
                
                # Unpack raw response safely whether returned as dict or string
                if isinstance(raw_response, dict):
                    answer_text = raw_response.get("answer", str(raw_response))
                else:
                    answer_text = str(raw_response)

                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
        
