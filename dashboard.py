# app.py - All-in-One ARGO RAG Application for Streamlit Cloud
"""
Complete ARGO RAG system in a single Streamlit application.
Includes database connection, embeddings, query processing, and visualization.
"""

import streamlit as st
import os
import json
import re
import math
import logging
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from datetime import datetime, timedelta
import psycopg2
from sqlalchemy import create_engine, text
import hashlib
import time

# Configure logging to suppress warnings
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="ARGO RAG System",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-online { color: #28a745; }
    .status-offline { color: #dc3545; }
    .query-result {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Environment configuration
DATABASE_URL = os.getenv('DATABASE_URL')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Database connection
@st.cache_resource
def get_database_connection():
    """Initialize database connection with caching"""
    if not DATABASE_URL:
        st.error("DATABASE_URL not configured. Please add it to Streamlit secrets.")
        st.stop()
    
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_size=3,
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={
                "connect_timeout": 10,
                "sslmode": "require"
            }
        )
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

# Load embedding model
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model with caching"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        return model
    except Exception as e:
        st.warning(f"Could not load embedding model: {e}")
        return None

# Geographic region definitions
REGION_BOUNDS = {
    'arabian sea': {'lat_min': 10, 'lat_max': 25, 'lon_min': 50, 'lon_max': 75},
    'bay of bengal': {'lat_min': 5, 'lat_max': 22, 'lon_min': 80, 'lon_max': 100},
    'indian ocean': {'lat_min': -40, 'lat_max': 25, 'lon_min': 20, 'lon_max': 120},
    'atlantic': {'lat_min': -60, 'lat_max': 70, 'lon_min': -80, 'lon_max': 20},
    'pacific': {'lat_min': -60, 'lat_max': 70, 'lon_min': 120, 'lon_max': -70},
    'mediterranean': {'lat_min': 30, 'lat_max': 46, 'lon_min': -6, 'lon_max': 42},
    'equator': {'lat_min': -10, 'lat_max': 10, 'lon_min': -180, 'lon_max': 180},
    'tropical': {'lat_min': -23.5, 'lat_max': 23.5, 'lon_min': -180, 'lon_max': 180},
}

class GeographicExtractor:
    """Extract geographic constraints from query text"""
    
    @staticmethod
    def extract_region(query: str) -> Optional[Dict]:
        query_lower = query.lower()
        
        for region, bounds in REGION_BOUNDS.items():
            if region in query_lower:
                return bounds
        
        # Look for coordinate patterns
        coord_pattern = r'(\d+\.?\d*)[°]?([NS])\s*,?\s*(\d+\.?\d*)[°]?([EW])'
        match = re.search(coord_pattern, query)
        if match:
            lat_val, lat_dir, lon_val, lon_dir = match.groups()
            lat = float(lat_val) * (1 if lat_dir.upper() == 'N' else -1)
            lon = float(lon_val) * (1 if lon_dir.upper() == 'E' else -1)
            
            return {
                'lat_min': lat - 5, 'lat_max': lat + 5,
                'lon_min': lon - 5, 'lon_max': lon + 5
            }
        
        return None

class TemporalExtractor:
    """Extract temporal constraints from query text"""
    
    @staticmethod
    def extract_time_range(query: str) -> Optional[Dict]:
        query_lower = query.lower()
        
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
        
        return None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers"""
    R = 6371
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon/2) * math.sin(delta_lon/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

@st.cache_data(ttl=300)
def get_database_stats():
    """Get database statistics with caching"""
    engine = get_database_connection()
    
    try:
        with engine.connect() as conn:
            # Basic stats
            total_result = conn.execute(text("SELECT COUNT(*) FROM floats")).fetchone()
            total_profiles = total_result[0] if total_result else 0
            
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
                    COUNT(*) as count
                FROM floats 
                WHERE lat IS NOT NULL AND lon IS NOT NULL
                GROUP BY region
                ORDER BY count DESC
            """)).fetchall()
            
            # Date range
            date_result = conn.execute(text("""
                SELECT MIN(time_start) as earliest, MAX(time_start) as latest 
                FROM floats WHERE time_start IS NOT NULL
            """)).fetchone()
            
            # Variable distribution
            var_result = conn.execute(text("""
                SELECT 
                    jsonb_array_elements_text(variables_list) as variable, 
                    COUNT(*) as count
                FROM floats 
                WHERE variables_list IS NOT NULL 
                    AND variables_list != '[]'::jsonb
                    AND variables_list != 'null'::jsonb
                GROUP BY variable
                ORDER BY count DESC
                LIMIT 10
            """)).fetchall()
            
            return {
                "total_profiles": total_profiles,
                "geographic_distribution": [
                    {"region": row[0], "count": row[1]} for row in geo_result
                ],
                "date_range": {
                    "earliest": str(date_result[0]) if date_result and date_result[0] else None,
                    "latest": str(date_result[1]) if date_result and date_result[1] else None
                },
                "variables": [{"name": row[0], "count": row[1]} for row in var_result if row[0]]
            }
            
    except Exception as e:
        st.error(f"Error getting database stats: {e}")
        return {"total_profiles": 0, "geographic_distribution": [], "date_range": {}, "variables": []}

def perform_vector_search(query: str, k: int = 5, geo_bounds=None, temp_constraints=None):
    """Perform vector similarity search"""
    engine = get_database_connection()
    model = load_embedding_model()
    
    if not model:
        # Fallback to text search if embeddings not available
        return perform_text_search(query, k, geo_bounds, temp_constraints)
    
    try:
        # Generate query embedding
        query_embedding = model.encode([query])[0]
        
        # Build SQL query
        base_sql = """
            SELECT 
                profile_id, 
                float_id, 
                lat, 
                lon, 
                time_start, 
                n_levels,
                variables_list, 
                parquet_path, 
                source_file,
                1 - (embedding <=> %s::vector) as similarity_score
            FROM floats 
            WHERE embedding IS NOT NULL
        """
        
        params = [query_embedding.tolist()]
        
        # Add geographic filter
        if geo_bounds:
            base_sql += " AND lat BETWEEN %s AND %s AND lon BETWEEN %s AND %s"
            params.extend([geo_bounds['lat_min'], geo_bounds['lat_max'], 
                          geo_bounds['lon_min'], geo_bounds['lon_max']])
        
        # Add temporal filter
        if temp_constraints:
            if 'months_back' in temp_constraints:
                base_sql += " AND time_start >= NOW() - INTERVAL '%s months'"
                params.append(temp_constraints['months_back'])
            elif 'year' in temp_constraints:
                base_sql += " AND EXTRACT(YEAR FROM time_start) = %s"
                params.append(temp_constraints['year'])
        
        base_sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding.tolist(), k])
        
        with engine.connect() as conn:
            result = conn.execute(text(base_sql), params)
            rows = result.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "profile_id": row[0],
                    "metadata": {
                        "float_id": row[1],
                        "lat": float(row[2]) if row[2] is not None else None,
                        "lon": float(row[3]) if row[3] is not None else None,
                        "time": str(row[4]) if row[4] else None,
                        "n_levels": int(row[5]) if row[5] else 0,
                        "variables_list": row[6] if row[6] else [],
                        "parquet": row[7],
                        "source_file": row[8]
                    },
                    "similarity_score": float(row[9]) if row[9] is not None else 0.0
                })
            
            return results
            
    except Exception as e:
        st.error(f"Vector search failed: {e}")
        return perform_text_search(query, k, geo_bounds, temp_constraints)

def perform_text_search(query: str, k: int = 5, geo_bounds=None, temp_constraints=None):
    """Fallback text search when vector search is not available"""
    engine = get_database_connection()
    
    try:
        # Build text search query
        search_terms = query.lower().split()
        
        base_sql = """
            SELECT 
                profile_id, 
                float_id, 
                lat, 
                lon, 
                time_start, 
                n_levels,
                variables_list, 
                parquet_path, 
                source_file,
                0.5 as similarity_score
            FROM floats 
            WHERE (
                LOWER(COALESCE(source_file, '')) LIKE %s
                OR LOWER(COALESCE(variables_list::text, '')) LIKE %s
                OR LOWER(COALESCE(float_id, '')) LIKE %s
            )
        """
        
        # Create search pattern
        search_pattern = f"%{' '.join(search_terms)}%"
        params = [search_pattern, search_pattern, search_pattern]
        
        # Add geographic filter
        if geo_bounds:
            base_sql += " AND lat BETWEEN %s AND %s AND lon BETWEEN %s AND %s"
            params.extend([geo_bounds['lat_min'], geo_bounds['lat_max'], 
                          geo_bounds['lon_min'], geo_bounds['lon_max']])
        
        # Add temporal filter
        if temp_constraints:
            if 'months_back' in temp_constraints:
                base_sql += " AND time_start >= NOW() - INTERVAL '%s months'"
                params.append(temp_constraints['months_back'])
            elif 'year' in temp_constraints:
                base_sql += " AND EXTRACT(YEAR FROM time_start) = %s"
                params.append(temp_constraints['year'])
        
        base_sql += " ORDER BY time_start DESC LIMIT %s"
        params.append(k)
        
        with engine.connect() as conn:
            result = conn.execute(text(base_sql), params)
            rows = result.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "profile_id": row[0],
                    "metadata": {
                        "float_id": row[1],
                        "lat": float(row[2]) if row[2] is not None else None,
                        "lon": float(row[3]) if row[3] is not None else None,
                        "time": str(row[4]) if row[4] else None,
                        "n_levels": int(row[5]) if row[5] else 0,
                        "variables_list": row[6] if row[6] else [],
                        "parquet": row[7],
                        "source_file": row[8]
                    },
                    "similarity_score": 0.5
                })
            
            return results
            
    except Exception as e:
        st.error(f"Text search failed: {e}")
        return []

def create_intelligent_response(query: str, results: List[Dict]) -> str:
    """Create intelligent response based on results"""
    if not results:
        return f"No oceanographic profiles found matching '{query}'. Try broader search terms or check if data has been ingested into the database."
    
    total_profiles = len(results)
    regions = set()
    variables = set()
    avg_similarity = np.mean([r.get('similarity_score', 0) for r in results])
    
    for r in results:
        metadata = r.get('metadata', {})
        lat = metadata.get('lat')
        lon = metadata.get('lon')
        
        if lat is not None and lon is not None:
            if 10 <= lat <= 25 and 50 <= lon <= 75:
                regions.add("Arabian Sea")
            elif -10 <= lat <= 10:
                regions.add("equatorial region")
            elif lat > 0:
                regions.add("Northern Hemisphere")
            else:
                regions.add("Southern Hemisphere")
        
        var_list = metadata.get('variables_list', [])
        if isinstance(var_list, list):
            variables.update([v.upper() for v in var_list if v])
    
    response_parts = [f"Found {total_profiles} oceanographic profiles"]
    
    if regions:
        response_parts.append(f"in the {', '.join(regions)}")
    
    if variables:
        param_names = []
        if 'TEMP' in variables:
            param_names.append("temperature")
        if 'PSAL' in variables:
            param_names.append("salinity")
        if 'DOXY' in variables:
            param_names.append("oxygen")
        if param_names:
            response_parts.append(f"measuring {', '.join(param_names)}")
    
    response_parts.append(f"with {avg_similarity:.2f} average relevance score.")
    
    return " ".join(response_parts)

def ingest_sample_data():
    """Ingest sample ARGO profile data"""
    engine = get_database_connection()
    model = load_embedding_model()
    
    sample_profiles = [
        {
            "profile_id": "sample_arabian_001",
            "float_id": "2901234",
            "lat": 18.5,
            "lon": 62.3,
            "time_start": "2023-03-15T10:30:00Z",
            "n_levels": 45,
            "variables_list": ["TEMP", "PSAL", "PRES"],
            "source_file": "sample_arabian.nc",
            "summary": "ARGO profile sample_arabian_001 from float 2901234 located at 18.5°N 62.3°E in Arabian Sea measured in March 2023 with temperature, salinity and pressure measurements."
        },
        {
            "profile_id": "sample_equatorial_002", 
            "float_id": "2901235",
            "lat": 2.1,
            "lon": 95.7,
            "time_start": "2023-04-20T14:15:00Z",
            "n_levels": 52,
            "variables_list": ["TEMP", "PSAL", "DOXY", "PRES"],
            "source_file": "sample_equatorial.nc",
            "summary": "ARGO profile sample_equatorial_002 from float 2901235 located at 2.1°N 95.7°E in equatorial waters measured in April 2023 with temperature, salinity, oxygen and pressure measurements."
        },
        {
            "profile_id": "sample_atlantic_003",
            "float_id": "2901236", 
            "lat": 25.8,
            "lon": -45.2,
            "time_start": "2023-02-10T08:45:00Z",
            "n_levels": 38,
            "variables_list": ["TEMP", "PSAL", "PRES"],
            "source_file": "sample_atlantic.nc",
            "summary": "ARGO profile sample_atlantic_003 from float 2901236 located at 25.8°N -45.2°E in Atlantic Ocean measured in February 2023 with temperature, salinity and pressure measurements."
        }
    ]
    
    try:
        with engine.connect() as conn:
            for profile in sample_profiles:
                # Generate embedding if model available
                embedding = None
                if model:
                    try:
                        embedding = model.encode([profile['summary']])[0].tolist()
                    except:
                        pass
                
                # Insert profile
                if embedding:
                    conn.execute(text("""
                        INSERT INTO floats (
                            profile_id, float_id, lat, lon, time_start, n_levels,
                            variables_list, source_file, embedding
                        ) VALUES (
                            :profile_id, :float_id, :lat, :lon, :time_start, :n_levels,
                            :variables_list, :source_file, :embedding
                        ) ON CONFLICT (profile_id) DO NOTHING
                    """), {
                        **{k: v for k, v in profile.items() if k != 'summary'},
                        "variables_list": json.dumps(profile['variables_list']),
                        "embedding": embedding
                    })
                else:
                    conn.execute(text("""
                        INSERT INTO floats (
                            profile_id, float_id, lat, lon, time_start, n_levels,
                            variables_list, source_file
                        ) VALUES (
                            :profile_id, :float_id, :lat, :lon, :time_start, :n_levels,
                            :variables_list, :source_file
                        ) ON CONFLICT (profile_id) DO NOTHING
                    """), {
                        **{k: v for k, v in profile.items() if k != 'summary'},
                        "variables_list": json.dumps(profile['variables_list'])
                    })
            
            conn.commit()
            return True
            
    except Exception as e:
        st.error(f"Failed to ingest sample data: {e}")
        return False

# Main Application
def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 ARGO RAG System</h1>', unsafe_allow_html=True)
    
    # Initialize database connection
    engine = get_database_connection()
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Query Interface")
        
        # System status
        with st.expander("📊 System Status", expanded=True):
            try:
                stats = get_database_stats()
                
                if stats["total_profiles"] > 0:
                    st.markdown('<span class="status-online">✅ Database Online</span>', unsafe_allow_html=True)
                    st.metric("Total Profiles", stats["total_profiles"])
                    
                    # Show date range
                    date_range = stats["date_range"]
                    if date_range.get("earliest") and date_range.get("latest"):
                        st.write(f"**Data Range:** {date_range['earliest'][:10]} to {date_range['latest'][:10]}")
                else:
                    st.markdown('<span class="status-offline">❌ No Data Found</span>', unsafe_allow_html=True)
                    if st.button("📥 Load Sample Data"):
                        with st.spinner("Loading sample data..."):
                            if ingest_sample_data():
                                st.success("Sample data loaded!")
                                st.experimental_rerun()
                            else:
                                st.error("Failed to load sample data")
                
                # Embedding status
                model = load_embedding_model()
                if model:
                    st.markdown('<span class="status-online">✅ Embeddings Ready</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-offline">⚠️ Text Search Only</span>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.markdown('<span class="status-offline">❌ Database Error</span>', unsafe_allow_html=True)
                st.error(f"Connection failed: {e}")
        
        st.divider()
        
        # Query input
        query = st.text_input(
            "Ask about ARGO data:", 
            placeholder="e.g., 'temperature profiles in Arabian Sea'",
            help="Use natural language to search oceanographic data"
        )
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            k = st.slider("Results", 1, 20, 5)
        with col2:
            search_mode = st.selectbox("Search", ["Smart", "Geographic", "Temporal"])
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            geographic_filter = st.checkbox("Geographic filtering", value=True)
            temporal_filter = st.checkbox("Temporal filtering", value=True)
            show_debug = st.checkbox("Show debug info", value=False)
        
        # Query button
        ask_button = st.button("🚀 Search", type="primary", use_container_width=True)
        
        st.divider()
        
        # Quick stats
        if 'stats' in locals() and stats["geographic_distribution"]:
            st.subheader("🌍 Data Distribution")
            geo_df = pd.DataFrame(stats["geographic_distribution"])
            if not geo_df.empty:
                st.dataframe(geo_df, use_container_width=True, hide_index=True)
    
    # Main content
    if ask_button and query:
        with st.spinner("🔍 Searching database..."):
            try:
                # Extract constraints
                geo_bounds = GeographicExtractor.extract_region(query) if geographic_filter else None
                temp_constraints = TemporalExtractor.extract_time_range(query) if temporal_filter else None
                
                # Perform search
                if geo_bounds or temp_constraints or search_mode != "Smart":
                    results = perform_vector_search(query, k, geo_bounds, temp_constraints)
                else:
                    results = perform_vector_search(query, k)
                
                # Store results
                st.session_state["last_results"] = {
                    "query": query,
                    "results": results,
                    "geo_bounds": geo_bounds,
                    "temp_constraints": temp_constraints,
                    "timestamp": datetime.now()
                }
                
                st.success(f"✅ Found {len(results)} profiles")
                
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
    
    # Display results
    if "last_results" in st.session_state:
        data = st.session_state["last_results"]
        results = data["results"]
        query = data["query"]
        
        st.subheader(f"🔍 Results for: '{query}'")
        
        if results:
            # AI Response
            with st.container():
                st.subheader("🤖 Analysis")
                answer = create_intelligent_response(query, results)
                st.markdown(f'<div class="query-result">{answer}</div>', unsafe_allow_html=True)
            
            # Results table
            st.subheader("📋 Profile Details")
            df_data = []
            for r in results:
                metadata = r.get("metadata", {})
                variables = metadata.get("variables_list", [])
                if isinstance(variables, str):
                    try:
                        variables = json.loads(variables)
                    except:
                        variables = []
                
                df_data.append({
                    "Profile ID": r["profile_id"],
                    "Float ID": metadata.get("float_id", "N/A"),
                    "Latitude": metadata.get("lat"),
                    "Longitude": metadata.get("lon"),
                    "Time": metadata.get("time", "N/A"),
                    "Variables": ", ".join(variables) if variables else "N/A",
                    "Levels": metadata.get("n_levels", "N/A"),
                    "Similarity": f"{r.get('similarity_score', 0):.3f}"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Latitude": st.column_config.NumberColumn(format="%.4f"),
                    "Longitude": st.column_config.NumberColumn(format="%.4f"),
                    "Similarity": st.column_config.NumberColumn(format="%.3f"),
                }
            )
            
            # Map visualization
            map_df = df[["Latitude", "Longitude"]].dropna()
            if not map_df.empty:
                st.subheader("🗺️ Geographic Distribution")
                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df.rename(columns={"Latitude": "lat", "Longitude": "lon"}),
                    get_position=["lon", "lat"],
                    get_fill_color=[255, 100, 100, 200],
                    get_radius=50000,
                    pickable=True,
                    filled=True
                )
                
                view = pdk.ViewState(
                    latitude=map_df["Latitude"].mean(),
                    longitude=map_df["Longitude"].mean(),
                    zoom=2,
                    pitch=0
                )
                
                st.pydeck_chart(pdk.Deck(
                    layers=[layer], 
                    initial_view_state=view,
                    tooltip={"text": "Profile: {Profile ID}\nLat: {lat}\nLon: {lon}"}
                ))
            
            # Profile details
            st.subheader("📊 Profile Inspector")
            profile_options = [""] + df["Profile ID"].tolist()
            selected_profile = st.selectbox(
                "Select profile for details:", 
                options=profile_options
            )
            
            if selected_profile:
                selected_result = next((r for r in results if r["profile_id"] == selected_profile), None)
                if selected_result:
                    metadata = selected_result.get("metadata", {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Profile Information:**")
                        st.write(f"• **Float ID**: {metadata.get('float_id', 'N/A')}")
                        st.write(f"• **Location**: {metadata.get('lat', 'N/A'):.4f}°N, {metadata.get('lon', 'N/A'):.4f}°E")
                        st.write(f"• **Time**: {metadata.get('time', 'N/A')}")
                        st.write(f"• **Depth Levels**: {metadata.get('n_levels', 'N/A')}")
                        st.write(f"• **Similarity**: {selected_result.get('similarity_score', 0):.3f}")
                    
                    with col2:
                        st.write("**Variables Measured:**")
                        variables = metadata.get('variables_list', [])
                        if isinstance(variables, str):
                            try:
                                variables = json.loads(variables)
                            except:
                                variables = []
                        
                        if variables:
                            for var in variables:
                                if var.upper() == 'TEMP':
                                    st.write("🌡️ Temperature")
                                elif var.upper() == 'PSAL':
                                    st.write("🧂 Salinity")
                                elif var.upper() == 'PRES':
                                    st.write("📏 Pressure")
                                elif var.upper() == 'DOXY':
                                    st.write("🫧 Dissolved Oxygen")
                                else:
                                    st.write(f"📊 {var}")
                        else:
                            st.write("No variable information available")
                    
                    # Source file info
                    source_file = metadata.get('source_file')
                    if source_file:
                        st.write(f"**Source File**: {source_file}")
                    
                    # Parquet file info
                    parquet_path = metadata.get('parquet')
                    if parquet_path:
                        st.write(f"**Data File**: {parquet_path}")
                        st.info("Full profile visualization requires access to the parquet data files.")
                    else:
                        st.warning("No detailed measurement data available for this profile.")
            
            # Debug information
            if show_debug and 'data' in locals():
                with st.expander("🔧 Debug Information"):
                    st.write("**Query Analysis:**")
                    if data.get("geo_bounds"):
                        st.json(data["geo_bounds"])
                    if data.get("temp_constraints"):
                        st.json(data["temp_constraints"])
                    
                    st.write("**Raw Results Sample:**")
                    if results:
                        st.json(results[0])
        
        else:
            st.info("No profiles found matching your query. Try different search terms or check if data has been loaded.")
    
    # Footer section
    st.divider()
    
    # Two columns for footer
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ℹ️ About This System")
        st.write("""
        This integrated ARGO RAG system provides natural language search over oceanographic float data:
        
        • **Database**: Direct PostgreSQL connection with pgvector
        • **Search**: Vector similarity + geographic/temporal filtering
        • **Visualization**: Interactive maps and profile details
        • **Data**: ARGO float temperature, salinity, and biogeochemical measurements
        """)
    
    with col2:
        st.subheader("💡 Usage Tips")
        st.write("""
        **Example queries to try:**
        
        • "temperature profiles in Arabian Sea"
        • "salinity data from equatorial region"
        • "oxygen measurements from 2023"
        • "recent profiles with biogeochemical data"
        • "floats near 25°N 65°E"
        """)
    
    # System information
    with st.expander("🔧 System Information & Setup"):
        st.write("**Environment Configuration:**")
        
        # Check required environment variables
        env_status = []
        required_vars = ["DATABASE_URL"]
        optional_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
        
        for var in required_vars:
            if os.getenv(var):
                env_status.append(f"✅ {var}: Configured")
            else:
                env_status.append(f"❌ {var}: Missing (Required)")
        
        for var in optional_vars:
            if os.getenv(var):
                env_status.append(f"✅ {var}: Configured")
            else:
                env_status.append(f"⚪ {var}: Not set (Optional)")
        
        for status in env_status:
            st.write(status)
        
        st.write("""
        **Required Environment Variables for Streamlit Cloud:**
        
        Add these in your Streamlit app settings → Secrets:
        
        ```toml
        DATABASE_URL = "postgresql://postgres:password@db.project.supabase.co:5432/postgres?sslmode=require"
        SUPABASE_URL = "https://project.supabase.co"  # Optional
        SUPABASE_KEY = "your-anon-key"  # Optional
        ```
        
        **Database Setup:**
        1. Create Supabase project
        2. Enable pgvector extension: `CREATE EXTENSION vector;`
        3. Run the database schema (see documentation)
        4. Add sample data using the button above
        """)
        
        # Database schema button
        if st.button("📋 Show Database Schema"):
            st.code("""
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the floats table
CREATE TABLE IF NOT EXISTS floats (
    id SERIAL PRIMARY KEY,
    profile_id TEXT UNIQUE,
    float_id TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    time_start TIMESTAMP WITH TIME ZONE,
    n_levels INTEGER DEFAULT 0,
    variables_list JSONB DEFAULT '[]'::jsonb,
    parquet_path TEXT,
    source_file TEXT,
    embedding vector(384), -- For sentence-transformers embeddings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_floats_lat_lon ON floats (lat, lon);
CREATE INDEX IF NOT EXISTS idx_floats_time_start ON floats (time_start);
CREATE INDEX IF NOT EXISTS idx_floats_variables ON floats USING GIN (variables_list);
CREATE INDEX IF NOT EXISTS idx_floats_embedding ON floats 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """, language="sql")
    
    # Performance monitoring
    if st.button("📊 Performance Check"):
        with st.spinner("Running system diagnostics..."):
            try:
                engine = get_database_connection()
                model = load_embedding_model()
                
                # Database performance
                start_time = time.time()
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM floats"))
                    count = result.scalar()
                db_time = time.time() - start_time
                
                # Embedding performance
                embed_time = None
                if model:
                    start_time = time.time()
                    test_embedding = model.encode(["test query"])
                    embed_time = time.time() - start_time
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Database Query", f"{db_time:.2f}s", f"{count} records")
                with col2:
                    if embed_time:
                        st.metric("Embedding Speed", f"{embed_time:.2f}s", "384 dimensions")
                    else:
                        st.metric("Embeddings", "N/A", "Not loaded")
                with col3:
                    memory_info = "Available" if model else "Text search only"
                    st.metric("Search Mode", "Vector" if model else "Text", memory_info)
                        
            except Exception as e:
                st.error(f"Performance check failed: {e}")

if __name__ == "__main__":
    main()
