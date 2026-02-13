import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Multi-PDF RAG", layout="wide")
st.title("📚 Multi-PDF RAG (OpenAI + FAISS)")

# ------------------ OPENAI KEY ------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ------------------ SESSION STATE ------------------
if "documents" not in st.session_state:
    st.session_state.documents = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ------------------ PDF UPLOAD ------------------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state.documents = []  # reset on re-upload

    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()

            # ✅ Add source filename to metadata
            for doc in docs:
                doc.metadata["source_file"] = pdf.name

            st.session_state.documents.extend(docs)

    st.success(f"{len(uploaded_files)} PDFs loaded successfully.")

# ------------------ CREATE EMBEDDINGS ------------------
if st.button("🚀 Create Embeddings"):
    if not st.session_state.documents:
        st.warning("Please upload PDFs first.")
    else:
        with st.spinner("Creating embeddings..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(st.session_state.documents)

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

            st.session_state.vectorstore = FAISS.from_documents(
                chunks, embeddings
            )

            st.success(f"Embeddings created for {len(chunks)} chunks.")

# ------------------ QUERY SECTION ------------------
if st.session_state.vectorstore:
    st.markdown("---")
    st.subheader("🔎 Ask Questions")

    query = st.text_input("Ask something across all uploaded PDFs")

    # ✅ Submit Button Added
    if st.button("Submit Query"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0
                )

                docs = st.session_state.vectorstore.similarity_search(
                    query, k=4
                )

                # Build context
                context = "\n\n".join(d.page_content for d in docs)

                prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

                response = llm.invoke(prompt)

                st.subheader("✅ Answer")
                st.write(response.content)

                # ✅ Show Source with File + Page
                with st.expander("📄 Source References"):
                    for d in docs:
                        file_name = d.metadata.get("source_file", "Unknown File")
                        page_number = d.metadata.get("page", "Unknown Page")

                        st.markdown(
                            f"**📁 File:** {file_name}  \n"
                            f"**📄 Page:** {page_number}"
                        )
                        st.write(d.page_content)
                        st.markdown("---")
