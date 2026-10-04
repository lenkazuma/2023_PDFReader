import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from reader_core import (
    EmptyDocumentError,
    answer_question,
    build_index,
    extract_documents,
    file_fingerprint,
    split_documents,
    summarize,
)

MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"]
EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv()
st.set_page_config(page_title="PDFReader", page_icon="📄")
st.title("PDF & Word Reader ✨")


def default_api_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    except FileNotFoundError:
        return os.getenv("OPENAI_API_KEY", "")


def record_usage(cb) -> None:
    usage = st.session_state.setdefault("usage", {"input": 0, "output": 0})
    for model_usage in cb.usage_metadata.values():
        usage["input"] += model_usage.get("input_tokens", 0)
        usage["output"] += model_usage.get("output_tokens", 0)


with st.sidebar:
    st.title("🤗💬 LLM PDFReader App")
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=default_api_key(),
        help="Only kept in this browser session. Get one at https://platform.openai.com/api-keys",
    )
    model = st.selectbox("Model 👉", MODELS)
    with st.expander("Advanced"):
        chunk_size = st.slider("Chunk size", 300, 3000, 1000, step=100)
        top_k = st.slider("Excerpts per answer", 2, 10, 4)
    usage = st.session_state.get("usage")
    if usage:
        st.caption(f"Chat tokens this session: {usage['input']:,} in / {usage['output']:,} out")
    st.markdown(
        """
        ## About
        Built with [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/)
        and [OpenAI](https://platform.openai.com/docs/models).
        """
    )

uploaded_file = st.file_uploader("Upload your file", type=["pdf", "docx"])

if uploaded_file is None:
    st.info("Upload a PDF or Word document to get a summary and ask questions about it.")
    st.stop()

if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to continue.")
    st.stop()

data = uploaded_file.getvalue()
doc_key = f"{file_fingerprint(data)}:{chunk_size}"

if st.session_state.get("doc_key") != doc_key:
    try:
        with st.spinner("Reading and indexing the document..."):
            docs = extract_documents(data, uploaded_file.type, uploaded_file.name)
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=min(200, chunk_size // 5))
            index = build_index(chunks, OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key))
    except EmptyDocumentError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not process the file: {exc}")
        st.stop()
    st.session_state.update(doc_key=doc_key, chunks=chunks, index=index, summary=None, history=[])

llm = ChatOpenAI(model=model, temperature=0.3, api_key=api_key)

st.header("Here's a brief summary of your file:")
if st.session_state.summary is None:
    try:
        with st.spinner("Summarising..."), get_usage_metadata_callback() as cb:
            st.session_state.summary = summarize(st.session_state.chunks, llm)
        record_usage(cb)
    except Exception as exc:
        st.error(f"Summary failed: {exc}")
st.write(st.session_state.summary or "")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        st.caption("Sources: " + ", ".join(f"p. {page}" for page in turn["pages"]))

question = st.chat_input("Ask a question about your file")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."), get_usage_metadata_callback() as cb:
                result = answer_question(
                    question,
                    st.session_state.index,
                    llm,
                    history=[(t["question"], t["answer"]) for t in st.session_state.history],
                    k=top_k,
                )
            record_usage(cb)
        except Exception as exc:
            st.error(f"An error occurred: {exc}")
            st.stop()
        pages = sorted({doc.metadata.get("page", "?") for doc in result.sources}, key=str)
        st.markdown(result.text)
        st.caption("Sources: " + ", ".join(f"p. {page}" for page in pages))
        with st.expander("Show excerpts"):
            for doc in result.sources:
                st.markdown(f"**p. {doc.metadata.get('page', '?')}**")
                st.text(doc.page_content)
    st.session_state.history.append({"question": question, "answer": result.text, "pages": pages})

if st.session_state.history and st.button("Clear conversation"):
    st.session_state.history = []
    st.rerun()
