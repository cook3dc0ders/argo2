# rag_server_fixed.py
"""
Fixed RAG server with proper geographic filtering and free LLM.
Replaces the original rag_server.py to solve the geographic query issues.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from embeddings_utils import EmbeddingManager
from config import PG
import json
import re
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import math

app = FastAPI(title="Fixed Argo RAG Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Geographic regions with precise bounds
GEOGRAPHIC_REGIONS = {
    'arabian sea': {'lat_min': 10, 'lat_max': 25, 'lon_min': 50, 'lon_max': 75},
    'bay of bengal': {'lat_min': 5, 'lat_max': 22, 'lon_min': 80, 'lon_max': 100},
    'equator': {'lat_min': -10, 'lat_max': 10, 'lon_min': -180, 'lon_max': 180},
    'equatorial': {'lat_min': -5, 'lat_max': 5, 'lon_min': -180, 'lon_max': 180},
    'tropical': {'lat_min': -23.5, 'lat_max': 23.5, 'lon_min': -180, 'lon_max': 180},
    'atlantic': {'lat_min': -60, 'lat_max': 70, 'lon_min': -80, 'lon_max': 20},
    'pacific': {'lat_min': -60, 'lat_max': 70, 'lon_min': 120, 'lon_max': -70},
    'mediterranean': {'lat_min': 30, 'lat_max': 46, 'lon_min': -6, 'lon_max': 42},
    'indian ocean': {'lat_min': -40, 'lat_max': 25, 'lon_min': 20, 'lon_max': 120}
}

# Initialize components
try:
    emb = EmbeddingManager()
    print("Embedding manager initialized")
except Exception as e:
    print(f"Failed to initialize embedding manager: {e}")
    emb = None

def pg_engine():
    """Create PostgreSQL engine"""
    url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
    return create_engine(url, pool_pre_ping=True, pool_size=20, max_overflow=30)

class QueryRequest(BaseModel):
    query: str
    k: int = 5

def extract_geographic_region(query: str) -> Optional[Dict]:
    """Extract geographic constraints from query"""
    query_lower = query.lower()
    
    for region_name, bounds in GEOGRAPHIC_REGIONS.items():
        if region_name in query_lower:
            return bounds
    
    # Look for coordinate patterns
    coord_pattern = r'(\d+\.?\d*)[°]?\s*([NS])\s*,?\s*(\d+\.?\d*)[°]?\s*([EW])'
    match = re.search(coord_pattern, query)
    if match:
        lat_val, lat_dir, lon_val, lon_dir = match.groups()
        lat = float(lat_val) * (1 if lat_dir.upper() == 'N' else -1)
        lon = float(lon_val) * (1 if lon_dir.upper() == 'E' else -1)
        
        # Create 10-degree box around the point
        return {
            'lat_min': lat - 5, 'lat_max': lat + 5,
            'lon_min': lon - 5, 'lon_max': lon + 5
        }
    
    return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers"""
    R = 6371
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def apply_geographic_filtering(results: List[Dict], geo_bounds: Dict) -> List[Dict]:
    """Apply strict geographic filtering to results"""
    if not geo_bounds:
        return results
    
    filtered = []
    center_lat = (geo_bounds['lat_min'] + geo_bounds['lat_max']) / 2
    center_lon = (geo_bounds['lon_min'] + geo_bounds['lon_max']) / 2
    
    for result in results:
        metadata = result.get('metadata', {})
        lat = metadata.get('lat')
        lon = metadata.get('lon')
        
        # Skip if no coordinates
        if lat is None or lon is None:
            continue
        
        # Check if within bounds
        if (geo_bounds['lat_min'] <= lat <= geo_bounds['lat_max'] and 
            geo_bounds['lon_min'] <= lon <= geo_bounds['lon_max']):
            
            # Calculate relevance score
            distance = haversine_distance(center_lat, center_lon, lat, lon)
            result['geographic_distance_km'] = distance
            result['geographic_relevance'] = max(0, 1 - distance / 2000)  # Normalize over 2000km
            filtered.append(result)
    
    # Sort by geographic relevance
    filtered.sort(key=lambda x: x.get('geographic_relevance', 0), reverse=True)
    return filtered

def create_intelligent_response(query: str, results: List[Dict], geo_bounds: Dict = None) -> str:
    """Create intelligent response without paid LLM"""
    if not results:
        if geo_bounds:
            region_desc = f"region {geo_bounds['lat_min']}-{geo_bounds['lat_max']}°N, {geo_bounds['lon_min']}-{geo_bounds['lon_max']}°E"
            return f"No oceanographic profiles found in {region_desc} matching '{query}'. Try expanding your search area or using different search terms."
        else:
            return f"No profiles found matching '{query}'. Try terms like 'temperature', 'salinity', 'equator', or 'Atlantic Ocean'."
    
    # Analyze results
    regions = set()
    variables = set()
    depth_info = []
    
    for r in results:
        metadata = r.get('metadata', {})
        lat = metadata.get('lat')
        lon = metadata.get('lon')
        
        # Classify region
        if lat is not None and lon is not None:
            if 10 <= lat <= 25 and 50 <= lon <= 75:
                regions.add("Arabian Sea")
            elif 5 <= lat <= 22 and 80 <= lon <= 100:
                regions.add("Bay of Bengal")
            elif -10 <= lat <= 10:
                regions.add("equatorial waters")
            elif 30 <= lat <= 46 and -6 <= lon <= 42:
                regions.add("Mediterranean Sea")
            elif -80 <= lon <= 20:
                regions.add("Atlantic Ocean")
            elif 120 <= lon <= 180 or -180 <= lon <= -70:
                regions.add("Pacific Ocean")
        
        # Extract variables
        var_str = metadata.get('variables_str', '').lower()
        if 'temp' in var_str:
            variables.add("temperature")
        if 'psal' in var_str:
            variables.add("salinity")
        if 'doxy' in var_str:
            variables.add("dissolved oxygen")
        if 'chla' in var_str:
            variables.add("chlorophyll")
        if 'nitrate' in var_str:
            variables.add("nitrate")
        
        # Collect depth information
        n_levels = metadata.get('n_levels', 0)
        if n_levels > 0:
            depth_info.append(n_levels)
    
    # Build intelligent response
    response_parts = [f"Found {len(results)} oceanographic profiles"]
    
    if regions:
        if len(regions) == 1:
            response_parts.append(f"in the {list(regions)[0]}")
        else:
            response_parts.append(f"across {', '.join(sorted(regions))}")
    
    if variables:
        response_parts.append(f"with measurements of {', '.join(sorted(variables))}")
    
    if depth_info:
        avg_depth = np.mean(depth_info)
        max_depth = max(depth_info)
        response_parts.append(f"Profiles contain an average of {avg_depth:.0f} depth levels (up to {max_depth} levels)")
    
    if geo_bounds:
        response_parts.append(f"All results are within your specified region")
    
    response_parts.append("Click on map markers to see exact locations, then select a profile to view detailed depth measurements.")
    
    return " ".join(response_parts) + "."

@app.post("/query")
def fixed_query_endpoint(qr: QueryRequest):
    """
    Fixed query endpoint with proper geographic filtering
    """
    if not emb:
        raise HTTPException(status_code=500, detail="Embedding manager not initialized")
    
    # Extract geographic constraints
    geo_bounds = extract_geographic_region(qr.query)
    
    try:
        # Get vector search results with higher k for filtering
        search_k = min(qr.k * 4, 100) if geo_bounds else qr.k
        chroma_res = emb.query(qr.query, n_results=search_k)
        
        # Process results
        ids = chroma_res.get("ids", [[]])[0]
        metadatas = chroma_res.get("metadatas", [[]])[0]
        docs = chroma_res.get("documents", [[]])[0]
        distances = chroma_res.get("distances", [[]])[0]
        
        results = []
        for i, profile_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            summary = docs[i] if i < len(docs) else ""
            distance = distances[i] if i < len(distances) else None
            
            results.append({
                "profile_id": profile_id,
                "metadata": metadata,
                "summary": summary,
                "similarity_score": 1 - distance if distance is not None else None
            })
        
        # Apply geographic filtering
        if geo_bounds:
            results = apply_geographic_filtering(results, geo_bounds)
            # Limit to requested number after filtering
            results = results[:qr.k]
        
        # Generate response
        answer = create_intelligent_response(qr.query, results, geo_bounds)
        
        return {
            "results": results,
            "answer": answer,
            "query": qr.query,
            "total_results": len(results),
            "geographic_filtering": {
                "applied": bool(geo_bounds),
                "region_bounds": geo_bounds,
                "region_detected": extract_geographic_region(qr.query) is not None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/health")
def health_check():
    """Health check"""
    try:
        engine = pg_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM floats"))
            db_count = result.scalar()
        db_status = f"connected ({db_count} profiles)"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"
    
    emb_status = "ready" if emb else "not initialized"
    if emb:
        try:
            stats = emb.get_collection_stats()
            doc_count = stats.get('total_documents', 0)
            emb_status = f"ready ({doc_count} embeddings)"
        except:
            emb_status = "error"
    
    return {
        "status": "healthy" if "connected" in db_status and "ready" in emb_status else "partial",
        "database": db_status,
        "embeddings": emb_status,
        "llm": "free model (no API key required)",
        "geographic_filtering": "enabled",
        "supported_regions": list(GEOGRAPHIC_REGIONS.keys())
    }

@app.get("/stats")
def get_stats():
    """Get database statistics with geographic breakdown"""
    try:
        engine = pg_engine()
        with engine.connect() as conn:
            # Total count
            result = conn.execute(text("SELECT COUNT(*) as total_profiles FROM floats")).fetchone()
            total_profiles = result[0] if result else 0
            
            # Geographic distribution
            geo_result = conn.execute(text("""
                SELECT 
                    CASE 
                        WHEN lat BETWEEN 10 AND 25 AND lon BETWEEN 50 AND 75 THEN 'Arabian Sea'
                        WHEN lat BETWEEN 5 AND 22 AND lon BETWEEN 80 AND 100 THEN 'Bay of Bengal'
                        WHEN lat BETWEEN -10 AND 10 THEN 'Equatorial'
                        WHEN lat BETWEEN 30 AND 46 AND lon BETWEEN -6 AND 42 THEN 'Mediterranean'
                        WHEN lon BETWEEN -80 AND 20 THEN 'Atlantic Ocean'
                        WHEN lon BETWEEN 120 AND 180 OR lon BETWEEN -180 AND -70 THEN 'Pacific Ocean'
                        ELSE 'Other Region'
                    END as region,
                    COUNT(*) as count,
                    AVG(lat) as avg_lat,
                    AVG(lon) as avg_lon,
                    MIN(lat) as min_lat,
                    MAX(lat) as max_lat,
                    MIN(lon) as min_lon,
                    MAX(lon) as max_lon
                FROM floats 
                WHERE lat IS NOT NULL AND lon IS NOT NULL
                GROUP BY region
                ORDER BY count DESC
            """)).fetchall()
            
            return {
                "total_profiles": total_profiles,
                "geographic_distribution": [
                    {
                        "region": row[0], 
                        "count": row[1], 
                        "center": {"lat": float(row[2]) if row[2] else None, "lon": float(row[3]) if row[3] else None},
                        "bounds": {
                            "lat_min": float(row[4]) if row[4] else None, "lat_max": float(row[5]) if row[5] else None,
                            "lon_min": float(row[6]) if row[6] else None, "lon_max": float(row[7]) if row[7] else None
                        }
                    } 
                    for row in geo_result
                ],
                "supported_regions": list(GEOGRAPHIC_REGIONS.keys())
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Fixed ARGO RAG Server with Geographic Filtering...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)