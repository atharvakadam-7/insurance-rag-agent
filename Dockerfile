FROM python:3.11-slim

WORKDIR /code

# build-essential is needed for some chromadb/sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time. This is the whole point of
# baking it in: without this line, the first request after every cold start
# on Render's free tier re-downloads ~90MB from Hugging Face, which is slow
# and will make your demo look broken to anyone testing it live.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Build the vectorstore from whatever PDFs are in data/ at image build time.
# Render's free-tier disk is ephemeral — it does NOT persist across deploys —
# so the index has to be rebuilt into every image, not built once and kept.
# If you add/change PDFs later, you rebuild the image. There is no
# "update the running container" path on this setup.
RUN python ingest.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]

