ARGO RAG PoC
============

What this PoC does
- Ingest ARGO NetCDF profiles (simple parser) → Parquet
- Store metadata in PostgreSQL
- Index summaries into Chroma (local vector DB) using sentence-transformers
- Simple FastAPI-based RAG server (with optional OpenAI integration)
- Streamlit dashboard with map, profile plot, and chat query

Requirements
- Docker & Docker-compose (for Postgres)
- Python 3.10+
- ~8GB RAM minimum for model embeddings; more for large volume

Setup
1. Start Postgres:
   docker-compose up -d

2. Create DB schema:
   psql -h localhost -U argo_user -d argo_db -f schema.sql

3. Python env:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

4. Edit config.py if necessary (paths, OPENAI_API_KEY etc)

5. Ingest data:
   python ingest.py --dir /path/to/your/netcdf_dir --parquet-dir ./parquet_store

6. Start backend:
   uvicorn rag_server:app --reload --port 8000

7. Start dashboard:
   streamlit run dashboard.py

Notes on scaling to 400GB
- Use Dask/Xarray distributed to parse NetCDF in parallel.
- Store Parquet on S3. Use Athena/Trino or a data warehouse for SQL queries.
- Use a hosted vector DB (Pinecone/Weaviate/Chroma Cloud) or cluster FAISS for large-scale retrieval.
- Use chunking: split each profile's level arrays into manageable documents for embeddings.

Extending
- Use `argopy` for robust ARGO reads and QC handling.
- Add QC-filtering using ARGO flags.
- Add PostGIS spatial queries and nearest-neighbor float lookup.
- Add LLM prompt-engineering for safe SQL generation & explanation.


















Get-Content .\schema.sql | docker exec -i argo-rag-poc-postgres-1 psql -U argo_user -d argo_db
