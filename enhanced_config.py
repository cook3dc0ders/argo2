# enhanced_config.py
"""
Enhanced configuration for ARGO RAG system with open-source LLM support.
Optimized for handling large datasets (50-60GB) efficiently.
"""
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

# Embedding model - optimized for oceanographic data
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Alternative embedding models for better performance:
# - "sentence-transformers/all-mpnet-base-v2" (better quality, slower)
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
# - "BAAI/bge-small-en-v1.5" (good for scientific text)

# Ollama configuration (local open-source LLM)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Alternative Ollama models:
# - "llama3.2:1b" (fastest, least capable)
# - "llama3.2:3b" (good balance of speed/capability)
# - "llama3.1:8b" (better quality, slower)
# - "mistral:7b" (good for structured tasks)
# - "codellama:7b" (better for SQL generation)

# Batch processing settings for large datasets
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))  # Process files in batches
MAX_CONCURRENT_FILES = int(os.getenv("MAX_CONCURRENT_FILES", 4))  # Parallel processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))  # Database insert chunk size

# Memory management for large datasets
MAX_PROFILES_IN_MEMORY = int(os.getenv("MAX_PROFILES_IN_MEMORY", 10000))
ENABLE_PROGRESS_CHECKPOINTS = os.getenv("ENABLE_PROGRESS_CHECKPOINTS", "true").lower() == "true"
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", 1000))  # Save progress every N profiles

# Query optimization settings
MAX_VECTOR_SEARCH_RESULTS = int(os.getenv("MAX_VECTOR_SEARCH_RESULTS", 50))
MAX_SQL_PREFILTER_RESULTS = int(os.getenv("MAX_SQL_PREFILTER_RESULTS", 200))
DEFAULT_SEARCH_RADIUS_DEGREES = float(os.getenv("DEFAULT_SEARCH_RADIUS_DEGREES", 2.0))

# Performance tuning
POSTGRES_CONNECTION_POOL_SIZE = int(os.getenv("POSTGRES_CONNECTION_POOL_SIZE", 10))
POSTGRES_MAX_OVERFLOW = int(os.getenv("POSTGRES_MAX_OVERFLOW", 20))
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", 500))

# Optional OpenAI fallback (if user provides key later)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Data validation settings
STRICT_COORDINATE_VALIDATION = os.getenv("STRICT_COORDINATE_VALIDATION", "true").lower() == "true"
REQUIRE_MINIMUM_LEVELS = int(os.getenv("REQUIRE_MINIMUM_LEVELS", 5))  # Minimum depth levels for valid profile
MAX_PROFILE_AGE_YEARS = int(os.getenv("MAX_PROFILE_AGE_YEARS", 30))  # Ignore very old data

# File processing settings
SKIP_CORRUPTED_FILES = os.getenv("SKIP_CORRUPTED_FILES", "true").lower() == "true"
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 500))  # Skip extremely large files
SUPPORTED_EXTENSIONS = os.getenv("SUPPORTED_EXTENSIONS", "nc,NC,netcdf,NETCDF").split(",")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "./logs/argo_rag.log")

# Regional definitions for better geographic understanding
GEOGRAPHIC_REGIONS = {
    'equatorial_pacific': {'lat_min': -5, 'lat_max': 5, 'lon_min': -180, 'lon_max': -80},
    'equatorial_atlantic': {'lat_min': -5, 'lat_max': 5, 'lon_min': -50, 'lon_max': 20},
    'equatorial_indian': {'lat_min': -5, 'lat_max': 5, 'lon_min': 40, 'lon_max': 120},
    'arabian_sea': {'lat_min': 8, 'lat_max': 27, 'lon_min': 50, 'lon_max': 80},
    'bay_of_bengal': {'lat_min': 5, 'lat_max': 25, 'lon_min': 80, 'lon_max': 100},
    'mediterranean': {'lat_min': 30, 'lat_max': 46, 'lon_min': -6, 'lon_max': 37},
    'north_atlantic': {'lat_min': 40, 'lat_max': 70, 'lon_min': -80, 'lon_max': 20},
    'south_atlantic': {'lat_min': -60, 'lat_max': -20, 'lon_min': -70, 'lon_max': 20},
    'north_pacific': {'lat_min': 20, 'lat_max': 70, 'lon_min': 120, 'lon_max': -120},
    'south_pacific': {'lat_min': -60, 'lat_max': -20, 'lon_min': 120, 'lon_max': -70},
    'southern_ocean': {'lat_min': -70, 'lat_max': -45, 'lon_min': -180, 'lon_max': 180},
}

# Seasonal definitions
SEASONAL_MONTHS = {
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'autumn': [9, 10, 11],
    'fall': [9, 10, 11],
    'winter': [12, 1, 2]
}

# Parameter groupings for better query understanding
PARAMETER_GROUPS = {
    'core': ['TEMP', 'PSAL', 'PRES'],
    'bgc': ['DOXY', 'NITRATE', 'PHOSPHATE', 'SILICATE', 'PH', 'CHLA'],
    'physical': ['TEMP', 'PSAL', 'PRES', 'CNDC'],
    'biogeochemical': ['DOXY', 'NITRATE', 'PHOSPHATE', 'SILICATE', 'PH', 'CHLA', 'BBP', 'CDOM'],
    'carbon': ['PH', 'ALKALINITY', 'DIC', 'PCO2'],
    'nutrients': ['NITRATE', 'PHOSPHATE', 'SILICATE'],
    'optics': ['CHLA', 'BBP', 'CDOM', 'DOWNWELLING_PAR']
}

# Create directories if they don't exist
import os
from pathlib import Path

for directory in [PARQUET_DIR, CHROMA_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

if LOG_TO_FILE:
    Path(LOG_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)