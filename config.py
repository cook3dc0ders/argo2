import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Postgres configuration
PG = {
    "user": os.getenv("PG_USER", "argo_user"),
    "password": os.getenv("PG_PASS", "argo_pass"),
    "db": os.getenv("PG_DB", "argo_db"),
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", 5432)),
}

# Parquet storage (use forward slashes even on Windows)
PARQUET_DIR = os.getenv("PARQUET_DIR", "./parquet_store")

# Chroma directory (for vector DB persistence)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Optional OpenAI key for better NL->SQL / answer generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")