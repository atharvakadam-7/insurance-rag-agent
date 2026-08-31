from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, TOP_K


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def get_retriever(k: int = TOP_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def format_docs(docs) -> str:
    """Turn retrieved chunks into one context string, tagged by source.
    Tagging matters: without it the LLM can't tell you which policy a clause
    came from when you're comparing two documents."""
    parts = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[Doc {i} | {source} | page {page}]\n{d.page_content}")
    return "\n\n".join(parts)
