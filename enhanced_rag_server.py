# enhanced_rag_server.py
"""
Enhanced RAG server with geographic filtering and free LLM integration.
Optimized for 40-50GB datasets with intelligent query processing.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from embeddings_utils import EmbeddingManager
from config import PG
import json
import re
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from transformers import pipeline
import math
import os
from pathlib import Path

# Database configuration for Railway
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:ewDINJqsTPPORxhBaUgPbPJQqEKPWpWP@shuttle.proxy.rlwy.net:35260/railway")

# If DATABASE_URL uses postgres:// (Railway sometimes does this), convert to postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Parse the DATABASE_URL for the PG dictionary
from urllib.parse import urlparse
parsed = urlparse(DATABASE_URL)

PG = {
    'user': parsed.username,
    'password': parsed.password,
    'host': parsed.hostname,
    'port': parsed.port or 5432,
    'db': parsed.path[1:]  # Remove leading slash
}

# Other configurations
PARQUET_DIR = Path("./parquet_store")
CHROMA_DIR = Path("./chroma_db")

app = FastAPI(title="Enhanced Argo RAG Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize free LLM for text generation
try:
    # Use a free, lightweight model for text generation
    text_generator = pipeline(
        "text-generation", 
        model="microsoft/DialoGPT-medium",
        device=-1,  # Use CPU
        max_length=200
    )
    print("Free text generation model loaded")
except Exception as e:
    print(f"Could not load text generation model: {e}")
    text_generator = None

# Initialize components
try:
    emb = EmbeddingManager()
    print("Embedding manager initialized")
except Exception as e:
    print(f"Failed to initialize embedding manager: {e}")
    emb = None

def pg_engine():
    """Create PostgreSQL engine with optimizations for large datasets"""
    url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
    return create_engine(
        url, 
        pool_pre_ping=True,
        pool_size=20,  # Increased for large datasets
        max_overflow=30,
        pool_recycle=3600
    )

class QueryRequest(BaseModel):
    query: str
    k: int = 5
    geographic_filter: bool = True
    temporal_filter: bool = True

class GeographicExtractor:
    """Extract geographic regions and coordinates from queries"""
    
    REGION_BOUNDS = {
        'arabian sea': {'lat_min': 10, 'lat_max': 25, 'lon_min': 50, 'lon_max': 75},
        'bay of bengal': {'lat_min': 5, 'lat_max': 22, 'lon_min': 80, 'lon_max': 100},
        'indian ocean': {'lat_min': -40, 'lat_max': 25, 'lon_min': 20, 'lon_max': 120},
        'atlantic': {'lat_min': -60, 'lat_max': 70, 'lon_min': -80, 'lon_max': 20},
        'pacific': {'lat_min': -60, 'lat_max': 70, 'lon_min': 120, 'lon_max': -70},
        'mediterranean': {'lat_min': 30, 'lat_max': 46, 'lon_min': -6, 'lon_max': 42},
        'equator': {'lat_min': -10, 'lat_max': 10, 'lon_min': -180, 'lon_max': 180},
        'tropical': {'lat_min': -23.5, 'lat_max': 23.5, 'lon_min': -180, 'lon_max': 180},
        'north atlantic': {'lat_min': 0, 'lat_max': 70, 'lon_min': -80, 'lon_max': 20},
        'south atlantic': {'lat_min': -60, 'lat_max': 0, 'lon_min': -70, 'lon_max': 20}
    }
    
    @classmethod
    def extract_region(cls, query: str) -> Optional[Dict]:
        """Extract geographic region from query text"""
        query_lower = query.lower()
        
        for region, bounds in cls.REGION_BOUNDS.items():
            if region in query_lower:
                return bounds
        
        # Look for coordinate patterns
        coord_pattern = r'(\d+\.?\d*)[°]?([NS])\s*,?\s*(\d+\.?\d*)[°]?([EW])'
        match = re.search(coord_pattern, query)
        if match:
            lat_val, lat_dir, lon_val, lon_dir = match.groups()
            lat = float(lat_val) * (1 if lat_dir.upper() == 'N' else -1)
            lon = float(lon_val) * (1 if lon_dir.upper() == 'E' else -1)
            
            # Create 5-degree box around the point
            return {
                'lat_min': lat - 5, 'lat_max': lat + 5,
                'lon_min': lon - 5, 'lon_max': lon + 5
            }
        
        return None

class TemporalExtractor:
    """Extract temporal constraints from queries"""
    
    @classmethod
    def extract_time_range(cls, query: str) -> Optional[Dict]:
        """Extract time constraints from query"""
        query_lower = query.lower()
        
        # Look for relative time expressions
        if 'last 6 months' in query_lower or 'past 6 months' in query_lower:
            return {'months_back': 6}
        elif 'last year' in query_lower or 'past year' in query_lower:
            return {'months_back': 12}
        elif 'last 3 months' in query_lower:
            return {'months_back': 3}
        
        # Look for specific years
        year_pattern = r'(20\d{2})'
        years = re.findall(year_pattern, query)
        if years:
            return {'year': int(years[0])}
        
        # Look for seasons
        if any(season in query_lower for season in ['summer', 'winter', 'monsoon', 'spring']):
            return {'seasonal': True}
        
        return None

class EnhancedQueryProcessor:
    """Enhanced query processing with geographic and temporal intelligence"""
    
    def __init__(self):
        self.geo_extractor = GeographicExtractor()
        self.temp_extractor = TemporalExtractor()
    
    def process_query(self, query: str, k: int = 5) -> Dict:
        """Process query with intelligent filtering"""
        
        # Extract geographic constraints
        geo_bounds = self.geo_extractor.extract_region(query)
        temp_constraints = self.temp_extractor.extract_time_range(query)
        
        # Build SQL filter based on constraints
        sql_filters = []
        filter_params = {}
        
        if geo_bounds:
            sql_filters.append(
                "lat BETWEEN :lat_min AND :lat_max AND "
                "lon BETWEEN :lon_min AND :lon_max"
            )
            filter_params.update(geo_bounds)
        
        if temp_constraints:
            if 'months_back' in temp_constraints:
                sql_filters.append(
                    "time_start >= NOW() - INTERVAL ':months months'"
                )
                filter_params['months'] = temp_constraints['months_back']
            elif 'year' in temp_constraints:
                sql_filters.append(
                    "EXTRACT(YEAR FROM time_start) = :year"
                )
                filter_params['year'] = temp_constraints['year']
        
        return {
            'original_query': query,
            'geographic_bounds': geo_bounds,
            'temporal_constraints': temp_constraints,
            'sql_filters': sql_filters,
            'filter_params': filter_params
        }

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers"""
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon/2) * math.sin(delta_lon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def filter_results_geographically(results: List[Dict], geo_bounds: Dict, query: str) -> List[Dict]:
    """Filter and rank results by geographic relevance"""
    if not geo_bounds or not results:
        return results
    
    filtered_results = []
    center_lat = (geo_bounds['lat_min'] + geo_bounds['lat_max']) / 2
    center_lon = (geo_bounds['lon_min'] + geo_bounds['lon_max']) / 2
    
    for result in results:
        metadata = result.get('metadata', {})
        lat = metadata.get('lat')
        lon = metadata.get('lon')
        
        if lat is not None and lon is not None:
            # Check if within bounds
            if (geo_bounds['lat_min'] <= lat <= geo_bounds['lat_max'] and 
                geo_bounds['lon_min'] <= lon <= geo_bounds['lon_max']):
                
                # Calculate distance from region center for ranking
                distance = haversine_distance(center_lat, center_lon, lat, lon)
                result['geographic_distance'] = distance
                result['geographic_relevance'] = max(0, 1 - distance / 1000)  # Normalize
                filtered_results.append(result)
    
    # Sort by geographic relevance
    filtered_results.sort(key=lambda x: x.get('geographic_relevance', 0), reverse=True)
    
    return filtered_results

def generate_free_llm_response(query: str, results: List[Dict]) -> str:
    """Generate response using free text generation model"""
    if not text_generator or not results:
        return create_rule_based_response(query, results)
    
    try:
        # Create context from results
        context_parts = []
        for r in results[:3]:
            metadata = r.get('metadata', {})
            lat = metadata.get('lat', 'unknown')
            lon = metadata.get('lon', 'unknown')
            variables = metadata.get('variables_str', 'unknown parameters')
            
            context_parts.append(f"Profile at {lat}N, {lon}E measuring {variables}")
        
        context = "; ".join(context_parts)
        
        # Create prompt for free model
        prompt = f"Query: {query}\nData found: {context}\nResponse:"
        
        # Generate response
        generated = text_generator(prompt, max_length=150, num_return_sequences=1)
        response = generated[0]['generated_text'].split("Response:")[-1].strip()
        
        return response if response else create_rule_based_response(query, results)
        
    except Exception as e:
        print(f"Free LLM generation failed: {e}")
        return create_rule_based_response(query, results)

def create_rule_based_response(query: str, results: List[Dict]) -> str:
    """Create intelligent rule-based response"""
    if not results:
        return f"No oceanographic profiles found matching '{query}'. Try broader search terms or check if data covers your region of interest."
    
    # Analyze the results
    total_profiles = len(results)
    regions = set()
    variables = set()
    
    for r in results:
        metadata = r.get('metadata', {})
        lat = metadata.get('lat')
        lon = metadata.get('lon')
        
        # Determine region
        if lat is not None and lon is not None:
            if 10 <= lat <= 25 and 50 <= lon <= 75:
                regions.add("Arabian Sea")
            elif -10 <= lat <= 10:
                regions.add("Equatorial region")
            elif lat > 0:
                regions.add("Northern Hemisphere")
            else:
                regions.add("Southern Hemisphere")
        
        # Extract variables
        var_str = metadata.get('variables_str', '')
        if 'temp' in var_str.lower():
            variables.add("temperature")
        if 'psal' in var_str.lower():
            variables.add("salinity")
        if 'doxy' in var_str.lower():
            variables.add("oxygen")
    
    # Build response
    response_parts = [f"Found {total_profiles} oceanographic profiles"]
    
    if regions:
        response_parts.append(f"in {', '.join(regions)}")
    
    if variables:
        response_parts.append(f"measuring {', '.join(variables)}")
    
    response_parts.append("Select a profile below to view detailed depth measurements and geographic location.")
    
    return " ".join(response_parts) + "."

# Initialize enhanced query processor
query_processor = EnhancedQueryProcessor()

@app.post("/query")
def enhanced_query_endpoint(qr: QueryRequest):
    """
    Enhanced query endpoint with geographic and temporal filtering
    """
    if not emb:
        raise HTTPException(status_code=500, detail="Embedding manager not initialized")
    
    # Process query to extract constraints
    query_analysis = query_processor.process_query(qr.query, qr.k)
    
    # Step 1: Get initial vector search results (higher k for filtering)
    initial_k = min(qr.k * 3, 50)  # Get more results for filtering
    try:
        chroma_res = emb.query(qr.query, n_results=initial_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")
    
    # Extract vector search results
    ids = chroma_res.get("ids", [[]])[0]
    metadatas = chroma_res.get("metadatas", [[]])[0]
    docs = chroma_res.get("documents", [[]])[0]
    distances = chroma_res.get("distances", [[]])[0]
    
    # Convert to structured results
    vector_results = []
    for i, profile_id in enumerate(ids):
        metadata = metadatas[i] if i < len(metadatas) else {}
        summary = docs[i] if i < len(docs) else ""
        distance = distances[i] if i < len(distances) else None
        
        vector_results.append({
            "profile_id": profile_id,
            "metadata": metadata,
            "summary": summary,
            "similarity_score": 1 - distance if distance is not None else None
        })
    
    # Step 2: Apply geographic filtering if constraints detected
    if query_analysis['geographic_bounds'] and qr.geographic_filter:
        vector_results = filter_results_geographically(
            vector_results, 
            query_analysis['geographic_bounds'],
            qr.query
        )
    
    # Step 3: Apply database-level filtering for additional constraints
    if query_analysis['sql_filters']:
        try:
            engine = pg_engine()
            with engine.connect() as conn:
                # Build filtered SQL query
                profile_ids = [r['profile_id'] for r in vector_results]
                if profile_ids:
                    profile_ids_str = "', '".join(profile_ids)
                    
                    base_sql = f"""
                        SELECT profile_id, float_id, lat, lon, time_start, n_levels, 
                               variables_list, parquet_path
                        FROM floats 
                        WHERE profile_id IN ('{profile_ids_str}')
                    """
                    
                    if query_analysis['sql_filters']:
                        base_sql += " AND " + " AND ".join(query_analysis['sql_filters'])
                    
                    # Execute filtered query
                    result = conn.execute(text(base_sql), query_analysis['filter_params'])
                    db_results = result.fetchall()
                    
                    # Update results with database information
                    db_profile_ids = {row[0] for row in db_results}
                    vector_results = [r for r in vector_results if r['profile_id'] in db_profile_ids]
        
        except Exception as e:
            print(f"Database filtering error: {e}")
            # Continue without database filtering
    
    # Step 4: Limit to requested number of results
    final_results = vector_results[:qr.k]
    
    # Step 5: Generate enhanced response
    if final_results:
        answer = generate_free_llm_response(qr.query, final_results)
    else:
        # No results found - provide helpful guidance
        answer = f"No profiles found for '{qr.query}'. "
        if query_analysis['geographic_bounds']:
            bounds = query_analysis['geographic_bounds']
            answer += f"Searched in region: {bounds['lat_min']}-{bounds['lat_max']}°N, {bounds['lon_min']}-{bounds['lon_max']}°E. "
        answer += "Try broader search terms or check if data exists for your region."
    
    return {
        "results": final_results,
        "answer": answer,
        "query": qr.query,
        "total_results": len(final_results),
        "query_analysis": query_analysis,
        "filtering_applied": {
            "geographic": bool(query_analysis['geographic_bounds'] and qr.geographic_filter),
            "temporal": bool(query_analysis['temporal_constraints'] and qr.temporal_filter)
        }
    }

@app.post("/smart_query")
def smart_query_endpoint(qr: QueryRequest):
    """
    Smart query that automatically determines best search strategy
    """
    query_lower = qr.query.lower()
    
    # Determine if this is a specific geographic query
    if any(region in query_lower for region in GeographicExtractor.REGION_BOUNDS.keys()):
        # Geographic query - use enhanced filtering
        return enhanced_query_endpoint(qr)
    else:
        # General query - use broader search
        qr.geographic_filter = False
        qr.temporal_filter = False
        return enhanced_query_endpoint(qr)

@app.get("/regions")
def get_supported_regions():
    """Get list of supported geographic regions"""
    return {
        "regions": list(GeographicExtractor.REGION_BOUNDS.keys()),
        "region_bounds": GeographicExtractor.REGION_BOUNDS
    }

@app.get("/stats")
def get_enhanced_stats():
    """Get enhanced database statistics including geographic distribution"""
    try:
        engine = pg_engine()
        with engine.connect() as conn:
            # Basic stats
            result = conn.execute(text("SELECT COUNT(*) as total_profiles FROM floats")).fetchone()
            total_profiles = result[0] if result else 0
            
            # Geographic distribution
            geo_result = conn.execute(text("""
                SELECT 
                    CASE 
                        WHEN lat BETWEEN 10 AND 25 AND lon BETWEEN 50 AND 75 THEN 'Arabian Sea'
                        WHEN lat BETWEEN -10 AND 10 THEN 'Equatorial'
                        WHEN lat BETWEEN 30 AND 46 AND lon BETWEEN -6 AND 42 THEN 'Mediterranean'
                        WHEN lon BETWEEN -80 AND 20 THEN 'Atlantic'
                        WHEN lon BETWEEN 120 AND 180 OR lon BETWEEN -180 AND -70 THEN 'Pacific'
                        ELSE 'Other'
                    END as region,
                    COUNT(*) as count,
                    AVG(lat) as avg_lat,
                    AVG(lon) as avg_lon
                FROM floats 
                WHERE lat IS NOT NULL AND lon IS NOT NULL
                GROUP BY region
                ORDER BY count DESC
            """)).fetchall()
            
            # Variable distribution
            var_result = conn.execute(text("""
                SELECT 
                    jsonb_array_elements_text(variables_list) as variable, 
                    COUNT(*) as count
                FROM floats 
                WHERE variables_list IS NOT NULL AND variables_list != '[]'::jsonb
                GROUP BY variable
                ORDER BY count DESC
            """)).fetchall()
            
            # Date range
            date_result = conn.execute(text("""
                SELECT MIN(time_start) as earliest, MAX(time_start) as latest 
                FROM floats WHERE time_start IS NOT NULL
            """)).fetchone()
            
            return {
                "total_profiles": total_profiles,
                "geographic_distribution": [
                    {"region": row[0], "count": row[1], "avg_lat": row[2], "avg_lon": row[3]} 
                    for row in geo_result
                ],
                "variables": [{"name": row[0], "count": row[1]} for row in var_result],
                "date_range": {
                    "earliest": str(date_result[0]) if date_result and date_result[0] else None,
                    "latest": str(date_result[1]) if date_result and date_result[1] else None
                },
                "supported_regions": list(GeographicExtractor.REGION_BOUNDS.keys())
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting enhanced stats: {str(e)}")

@app.get("/health")
def health_check():
    """Enhanced health check"""
    # Check database
    try:
        engine = pg_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM floats"))
            db_count = result.scalar()
        db_status = f"connected ({db_count} profiles)"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"
    
    # Check embeddings
    emb_status = "ready" if emb else "not initialized"
    if emb:
        try:
            stats = emb.get_collection_stats()
            doc_count = stats.get('total_documents', 0)
            emb_status = f"ready ({doc_count} embeddings)"
        except:
            emb_status = "error"
    
    # Check LLM
    llm_status = "free model ready" if text_generator else "no text generation"
    
    return {
        "status": "healthy" if "connected" in db_status and "ready" in emb_status else "partial",
        "database": db_status,
        "embeddings": emb_status,
        "llm": llm_status,
        "geographic_filtering": "enabled",
        "temporal_filtering": "enabled"
    }

if __name__ == "__main__":
    import uvicorn
    import os
    print("Starting Enhanced ARGO RAG Server...")
    print("Features: Geographic filtering, Free LLM, Large dataset optimization")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)

