"""
Builds (or rebuilds) the Chroma vectorstore from PDFs in data/.

Run this locally whenever you add/change PDFs. It also runs once at Docker
build time (see Dockerfile) so the image ships with a ready index — but if
you change the PDFs after that, you must rebuild the image, since Render's
free-tier disk doesn't persist across deploys.
"""
import os
import shutil
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.rag import get_embeddings
from app.config import CHROMA_PERSIST_DIR

DATA_DIR = "data"
MIN_CHUNK_CHARS = 50  # drops title-only chunks that dominate retrieval


def clean_text(text: str) -> str:
    """Cleans common PDF extraction artifacts, corrupt math symbols, and encoding glitches."""
    if not text:
        return ""

    # Fix corrupt symbol artifacts common in insurance PDFs
    text = text.replace("â¯â¥â¯", " >= ")
    text = text.replace("â¥", " >= ")
    text = text.replace("â¤", " <= ")
    text = text.replace("â¯", " ")
    text = text.replace("â", "-")
    text = text.replace("\xa0", " ")  # Non-breaking space

    # Normalize whitespace and redundant newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()


def load_documents():
    docs = []
    if not os.path.isdir(DATA_DIR):
        return docs
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(DATA_DIR, fname)
            loaded_docs = PyPDFLoader(path).load()

            # Clean text in loaded documents
            for doc in loaded_docs:
                doc.page_content = clean_text(doc.page_content)

            docs.extend(loaded_docs)
    return docs


def build_index():
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)  # kill stale embeddings, don't append to them

    docs = load_documents()
    if not docs:
        print(f"No PDFs found in {DATA_DIR}/. Add policy PDFs, then re-run this script.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]

    if not chunks:
        print("All chunks were filtered out as too short — check your PDFs aren't scanned images.")
        return

    Chroma.from_documents(chunks, get_embeddings(), persist_directory=CHROMA_PERSIST_DIR)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} pages into {CHROMA_PERSIST_DIR}/")

    # DIAGNOSTIC: confirm the cleanup actually took, right after rebuild.
    # If this prints artifacts (â¯, â¥, â), clean_text isn't touching the
    # text that's actually being embedded — check DATA_DIR and file encoding.
    sample = chunks[0].page_content[:200]
    print(f"Sample cleaned chunk: {sample!r}")


if __name__ == "__main__":
    build_index()