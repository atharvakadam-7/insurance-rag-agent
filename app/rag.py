from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from .config import CHROMA_PERSIST_DIR, TOP_K

# FastEmbed uses ONNX runtime instead of torch/sentence-transformers.
# Same idea as before (small local embedding model) but without the ~700MB
# torch runtime that was causing the 512MB Render instance to OOM.
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings():
    return FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)


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