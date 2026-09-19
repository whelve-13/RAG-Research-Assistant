import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    """Combines retrieved documents into formatted text."""
    return "\n\n".join(doc.page_content for doc in docs)

def process_pdf(file_path: str):
    """Parses a PDF, chunks text, and builds a local vector index."""
    # 1. Load document pages
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Chunk text with overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)

    # 3. Local embedding model (runs free on CPU)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Store chunks in Chroma
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

class RAGPipeline:
    def __init__(self, retriever, llm_chain):
        self.retriever = retriever
        self.llm_chain = llm_chain

    def invoke(self, inputs: dict):
        query = inputs["input"]
        # Fetch matching source documents
        docs = self.retriever.invoke(query)
        # Generate response using context
        answer = self.llm_chain.invoke({"context": format_docs(docs), "input": query})
        return {"answer": answer, "context": docs}

def build_rag_chain(retriever, groq_api_key: str):
    """Connects Chroma retrieval to Groq LLM using native LCEL."""
    os.environ["GROQ_API_KEY"] = groq_api_key

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.1
    )

    system_prompt = (
        "You are an academic research assistant. Use the following retrieved "
        "context to answer the user's question clearly and accurately. "
        "If the answer cannot be found in the context, explicitly state that you do not know. "
        "Do not fabricate facts.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm_chain = prompt | llm | StrOutputParser()
    return RAGPipeline(retriever, llm_chain)