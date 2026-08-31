import os
from dotenv import load_dotenv

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Check Groq's console for current model names — they deprecate models
# without much warning. This was correct at time of writing.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
print(f"[startup] Using GROQ_MODEL={GROQ_MODEL}")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
TOP_K = int(os.getenv("TOP_K", "4"))


def require_groq_key():
    """Called lazily (not at import time) so ingest.py and tests don't need
    a Groq key just to build the vectorstore."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
