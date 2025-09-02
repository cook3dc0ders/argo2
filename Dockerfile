FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p parquet_store chroma_db sample_data

# Generate sample data and set up database on startup
# (This will be handled by the startup command)

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:'+str(__import__('os').environ.get('PORT', '8000'))+'/health', timeout=5)" || exit 1

# Use the enhanced server
CMD ["sh", "-c", "python create_sample_data.py --output-dir sample_data --n-single 10 --n-multi 5 && python simple_database_setup.py && python ultra_robust_ingest.py --dir sample_data --parquet-dir parquet_store --verbose && uvicorn enhanced_rag_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
