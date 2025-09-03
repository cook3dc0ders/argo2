# cloud_api.py
"""
Lightweight FastAPI backend for cloud deployment.
Optimized for Railway/Render free tiers.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ARGO RAG API", 
    version="1.0.0",
    description="Lightweight API for ARGO oceanographic data"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    k: int = 5

def get_database_connection():
    """Get database connection from environment"""
    try:
        # Try DATABASE_URL first (Railway, Render)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            return psycopg2.connect(database_url)
        
        # Try individual components
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", 5432)),
            database=os.getenv("PG_DB", "argo_db"),
            user=os.getenv("PG_USER", "argo_user"),
            password=os.getenv("PG_PASS", "argo_pass")
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ARGO RAG API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": ["/health", "/query", "/stats"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_database_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM floats")
                count = cursor.fetchone()[0]
            conn.close()
            
            return {
                "status": "healthy",
                "database": f"connected ({count} profiles)",
                "embeddings": "not implemented in lite version",
                "openai": "not implemented in lite version"
            }
        else:
            return {
                "status": "degraded",
                "database": "not connected",
                "embeddings": "not available",
                "openai": "not available"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        conn = get_database_connection()
        if not conn:
            raise HTTPException(status_code=503, detail="Database not available")
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Total profiles
            cursor.execute("SELECT COUNT(*) as total_profiles FROM floats")
            total_profiles = cursor.fetchone()["total_profiles"]
            
            # Date range
            cursor.execute("""
                SELECT MIN(time_start) as earliest, MAX(time_start) as latest 
                FROM floats WHERE time_start IS NOT NULL
            """)
            date_range = cursor.fetchone()
            
            # Variable distribution
            cursor.execute("""
                SELECT 
                    jsonb_array_elements_text(variables_list) as variable, 
                    COUNT(*) as count
                FROM floats 
                WHERE variables_list IS NOT NULL 
                GROUP BY variable
                ORDER BY count DESC
                LIMIT 10
            """)
            variables = [{"name": row["variable"], "count": row["count"]} 
                        for row in cursor.fetchall()]
            
            # Geographic distribution (simplified)
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN lat BETWEEN -10 AND 10 THEN 'Equatorial'
                        WHEN lat > 40 THEN 'Northern High'
                        WHEN lat < -40 THEN 'Southern High'
                        WHEN lat > 0 THEN 'Northern Mid'
                        ELSE 'Southern Mid'
                    END as region,
                    COUNT(*) as count
                FROM floats 
                WHERE lat IS NOT NULL
                GROUP BY region
                ORDER BY count DESC
            """)
            regions = [{"region": row["region"], "count": row["count"]} 
                      for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "total_profiles": total_profiles,
            "date_range": {
                "earliest": str(date_range["earliest"]) if date_range["earliest"] else None,
                "latest": str(date_range["latest"]) if date_range["latest"] else None
            },
            "variables": variables,
            "regions": regions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/query")
async def query_profiles(request: QueryRequest):
    """Query ARGO profiles"""
    try:
        conn = get_database_connection()
        if not conn:
            raise HTTPException(status_code=503, detail="Database not available")
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Simple text search
            cursor.execute("""
                SELECT float_id, profile_id, lat, lon, time_start, n_levels, 
                       variables_list, parquet_path
                FROM floats 
                WHERE LOWER(CAST(variables_list AS TEXT)) LIKE LOWER(%s)
                   OR LOWER(float_id) LIKE LOWER(%s)
                   OR LOWER(profile_id) LIKE LOWER(%s)
                ORDER BY time_start DESC
                LIMIT %s
            """, (f"%{request.query}%", f"%{request.query}%", 
                  f"%{request.query}%", request.k))
            
            results = cursor.fetchall()
            
            formatted_results = []
            for row in results:
                try:
                    variables = json.loads(row["variables_list"]) if row["variables_list"] else []
                except:
                    variables = []
                
                formatted_results.append({
                    "profile_id": row["profile_id"],
                    "metadata": {
                        "float_id": row["float_id"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "time": str(row["time_start"]) if row["time_start"] else None,
                        "parquet": row["parquet_path"],
                        "variables": variables
                    },
                    "similarity_score": 0.8  # Mock similarity since we don't have embeddings
                })
        
        conn.close()
        
        return {
            "results": formatted_results,
            "answer": f"Found {len(formatted_results)} profiles matching '{request.query}'",
            "query": request.query,
            "total_results": len(formatted_results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
