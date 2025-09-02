FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p parquet_store chroma_db

# For free hosting, reduce startup complexity
CMD ["uvicorn", "enhanced_rag_server:app", "--host", "0.0.0.0", "--port", "$PORT"]
