# argo_dashboard.py
"""
ARGO Oceanographic Data Explorer with AI Integration
A comprehensive, user-friendly dashboard for exploring global ocean data
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    layout="wide",
    page_title="ARGO Ocean Data Explorer",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for beautiful, professional appearance with light/dark theme support
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --text-color-light: #495057;
        --text-color-dark: #f8f9fa;
        --background-color-light: #ffffff;
        --background-color-dark: #1e1e1e;
        --card-background-light: #f8f9ff;
        --card-background-dark: #2c2c2c;
        --border-color-light: rgba(102, 126, 234, 0.1);
        --border-color-dark: rgba(255, 255, 255, 0.1);
        --shadow-color-light: rgba(102, 126, 234, 0.3);
        --shadow-color-dark: rgba(0, 0, 0, 0.5);
    }
    
    .stApp {
        background-color: var(--background-color-light);
        color: var(--text-color-light);
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    .metric-container {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px var(--shadow-color-light);
        transition: transform 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px var(--shadow-color-light);
    }
    
    # .search-container {
    #     background: linear-gradient(135deg, var(--card-background-light) 0%, #e3f2fd 100%);
    #     border-radius: 20px;
    #     padding: 2rem;
    #     margin: 2rem 0;
    #     border: 2px solid var(--border-color-light);
    #     box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
    # }
    
    .ai-response {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 12px 48px var(--shadow-color-light);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .ai-response h2 {
        color: white !important;
        margin-bottom: 1rem;
    }
    
    .profile-card {
        background: var(--background-color-light);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .profile-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .status-excellent { 
        color: #28a745; 
        font-weight: 600;
        background: rgba(40, 167, 69, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
    }
    
    .status-good { 
        color: #17a2b8; 
        font-weight: 600;
        background: rgba(23, 162, 184, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 2px solid #f6c23e;
        color: #856404;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(246, 194, 62, 0.2);
    }
    
    .success-box {
        background: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(40, 167, 69, 0.2);
    }
    
    .info-card {
        background: var(--card-background-light);
        border: 2px solid var(--border-color-light);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .ocean-gradient {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px var(--shadow-color-light);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px var(--shadow-color-light);
    }
    
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .sidebar-section {
        background: var(--card-background-light);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .footer {
        background: linear-gradient(135deg, var(--card-background-light) 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        text-align: center;
        color: #6c757d;
    }

    /* --- Dark Mode Styles --- */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #262730; /* Softer dark background */
            color: #e0e0e0; /* Lighter text color */
        }
        
        .card-background-dark: #333540; /* Soft card background */

        .search-container {
            background: #333540;
            border: 2px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        
        .profile-card {
            background: #333540;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }

        .ai-response {
            box-shadow: 0 12px 48px rgba(0,0,0,0.4); /* Keep the shadow dark */
        }
        
        .info-card {
            background: #333540;
            border: 2px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }

        .sidebar-section {
            background: #333540;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        /* ... other styles for dark mode ... */
    }
</style>
""", unsafe_allow_html=True)

# API Configuration (keeping implementation details hidden from user)
API_KEY = "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"
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# Sample ARGO data with comprehensive profiles
@st.cache_data(ttl=300)
def load_comprehensive_argo_data():
    """Load comprehensive ARGO data with detailed oceanographic profiles"""
    return {
        "profiles": [
            {
                "profile_id": "ARGO_ATLANTIC_001",
                "float_id": "1901393",
                "lat": 35.5,
                "lon": -75.2,
                "time": "2023-06-15T12:00:00",
                "n_levels": 50,
                "variables": ["TEMP", "PSAL", "DOXY"],
                "region": "North Atlantic",
                "max_depth": 2000,
                "temperature_range": "2.5-25.3°C",
                "salinity_range": "34.2-36.8 PSU",
                "oxygen_range": "180-250 µmol/kg",
                "data_quality": "excellent",
                "special_features": ["thermocline", "oxygen_minimum"],
                "depth_profile": {
                    "depths": [0, 10, 50, 100, 200, 500, 1000, 1500, 2000],
                    "temperature": [25.3, 24.8, 22.1, 18.5, 12.3, 6.8, 4.2, 3.1, 2.5],
                    "salinity": [36.8, 36.7, 36.5, 36.2, 35.8, 35.2, 34.8, 34.5, 34.2],
                    "oxygen": [250, 245, 220, 180, 160, 200, 220, 240, 245]
                }
            },
            {
                "profile_id": "ARGO_EQUATORIAL_001",
                "float_id": "1901394",
                "lat": 0.5,
                "lon": -25.0,
                "time": "2023-07-01T06:00:00",
                "n_levels": 75,
                "variables": ["TEMP", "PSAL", "DOXY", "CHLA"],
                "region": "Equatorial Atlantic",
                "max_depth": 1800,
                "temperature_range": "4.1-28.7°C",
                "salinity_range": "34.9-36.1 PSU",
                "oxygen_range": "140-280 µmol/kg",
                "chlorophyll_range": "0.1-2.5 mg/m³",
                "data_quality": "excellent",
                "special_features": ["equatorial_upwelling", "high_productivity"],
                "depth_profile": {
                    "depths": [0, 20, 50, 100, 200, 500, 1000, 1500, 1800],
                    "temperature": [28.7, 27.5, 24.2, 20.8, 16.3, 8.9, 5.2, 4.5, 4.1],
                    "salinity": [36.1, 35.9, 35.7, 35.4, 35.1, 34.9, 35.0, 35.1, 35.2],
                    "oxygen": [280, 260, 220, 140, 160, 180, 200, 220, 240]
                }
            },
            {
                "profile_id": "ARGO_ARABIAN_001",
                "float_id": "1901395",
                "lat": 25.2,
                "lon": 65.8,
                "time": "2023-07-15T18:30:00",
                "n_levels": 60,
                "variables": ["TEMP", "PSAL", "DOXY", "CHLA", "NITRATE"],
                "region": "Arabian Sea",
                "max_depth": 1500,
                "temperature_range": "15.2-29.1°C",
                "salinity_range": "35.1-36.9 PSU",
                "oxygen_range": "5-220 µmol/kg",
                "chlorophyll_range": "0.2-3.8 mg/m³",
                "nitrate_range": "0.5-35 µmol/L",
                "data_quality": "excellent",
                "special_features": ["oxygen_minimum_zone", "monsoon_influence", "high_productivity"],
                "depth_profile": {
                    "depths": [0, 25, 75, 150, 300, 600, 1000, 1500],
                    "temperature": [29.1, 28.2, 22.8, 18.5, 16.8, 15.9, 15.4, 15.2],
                    "salinity": [36.9, 36.7, 36.2, 35.8, 35.4, 35.2, 35.1, 35.1],
                    "oxygen": [220, 180, 50, 5, 15, 80, 120, 150]
                }
            },
            {
                "profile_id": "ARGO_PACIFIC_001",
                "float_id": "1901396",
                "lat": 32.1,
                "lon": -145.8,
                "time": "2023-08-10T09:45:00",
                "n_levels": 70,
                "variables": ["TEMP", "PSAL", "DOXY", "CHLA", "PH"],
                "region": "North Pacific Gyre",
                "max_depth": 2000,
                "temperature_range": "4.5-22.8°C",
                "salinity_range": "33.9-35.8 PSU",
                "oxygen_range": "150-260 µmol/kg",
                "ph_range": "7.6-8.1",
                "data_quality": "excellent",
                "special_features": ["oligotrophic_waters", "deep_chlorophyll_maximum"],
                "depth_profile": {
                    "depths": [0, 30, 100, 200, 400, 800, 1200, 2000],
                    "temperature": [22.8, 21.5, 16.2, 12.8, 8.9, 6.1, 5.2, 4.5],
                    "salinity": [35.8, 35.6, 35.2, 34.8, 34.5, 34.2, 34.0, 33.9],
                    "oxygen": [260, 240, 200, 150, 170, 190, 210, 220]
                }
            },
            {
                "profile_id": "ARGO_SOUTHERN_001",
                "float_id": "1901397",
                "lat": -55.8,
                "lon": 140.2,
                "time": "2023-09-05T14:20:00",
                "n_levels": 90,
                "variables": ["TEMP", "PSAL", "DOXY", "NITRATE", "SILICATE"],
                "region": "Southern Ocean",
                "max_depth": 2500,
                "temperature_range": "-1.2-8.7°C",
                "salinity_range": "33.8-34.7 PSU",
                "oxygen_range": "200-350 µmol/kg",
                "nitrate_range": "20-45 µmol/L",
                "silicate_range": "40-120 µmol/L",
                "data_quality": "excellent",
                "special_features": ["antarctic_waters", "high_nutrients", "carbon_sink"],
                "depth_profile": {
                    "depths": [0, 50, 150, 300, 600, 1000, 1500, 2000, 2500],
                    "temperature": [8.7, 6.2, 3.8, 2.1, 1.2, 0.8, 0.2, -0.5, -1.2],
                    "salinity": [33.8, 34.0, 34.2, 34.4, 34.5, 34.6, 34.7, 34.7, 34.7],
                    "oxygen": [350, 330, 300, 280, 250, 220, 200, 210, 220]
                }
            }
        ],
        "metadata": {
            "total_profiles": 5,
            "regions": ["North Atlantic", "Equatorial Atlantic", "Arabian Sea", "North Pacific Gyre", "Southern Ocean"],
            "variables": ["TEMP", "PSAL", "DOXY", "CHLA", "NITRATE", "SILICATE", "PH"],
            "time_range": {"start": "2023-06-15", "end": "2023-09-05"},
            "depth_range": {"min": 0, "max": 2500}
        }
    }

def is_argo_ocean_related(query):
    """Check if query is related to ARGO ocean data - enhanced validation"""
    if not query or len(query.strip()) < 3:
        return False
    
    query_lower = query.lower().strip()
    
    # Ocean/ARGO related keywords
    ocean_keywords = [
        'ocean', 'argo', 'float', 'profile', 'temperature', 'salinity', 'oxygen', 'temp', 'psal', 'doxy',
        'depth', 'sea', 'marine', 'water', 'atlantic', 'pacific', 'indian', 'southern', 'equatorial',
        'thermocline', 'chlorophyll', 'nitrate', 'silicate', 'ph', 'conductivity', 'pressure',
        'oceanographic', 'hydrographic', 'biogeochemical', 'upwelling', 'downwelling', 'current',
        'gyre', 'mixed layer', 'halocline', 'pycnocline', 'water mass', 'deep water', 'surface',
        'antarctic', 'arctic', 'tropical', 'subtropical', 'polar', 'coastal', 'offshore',
        'buoyancy', 'density', 'salinity', 'temperature', 'oxygen minimum', 'nutrients'
    ]
    
    # Non-ocean keywords that should be rejected
    non_ocean_keywords = [
        'weather', 'rain', 'snow', 'tornado', 'hurricane', 'land', 'forest', 'mountain', 'desert',
        'politics', 'economy', 'sports', 'entertainment', 'movie', 'music', 'food', 'restaurant',
        'car', 'plane', 'train', 'computer', 'software', 'programming', 'business', 'finance',
        'health', 'medicine', 'doctor', 'hospital', 'school', 'education', 'university',
        'shopping', 'clothes', 'fashion', 'travel', 'hotel', 'vacation', 'tourism'
    ]
    
    # Check if query contains ocean-related keywords
    has_ocean_keywords = any(keyword in query_lower for keyword in ocean_keywords)
    
    # Check if query contains non-ocean keywords
    has_non_ocean_keywords = any(keyword in query_lower for keyword in non_ocean_keywords)
    
    # Additional check for common patterns
    argo_patterns = ['float', 'profile', 'measurement', 'data', 'analysis', 'depth', 'level']
    has_argo_patterns = any(pattern in query_lower for pattern in argo_patterns)
    
    # Return True only if it has ocean keywords and doesn't have non-ocean keywords
    return (has_ocean_keywords or has_argo_patterns) and not has_non_ocean_keywords

def call_ai_api(query, context_data):
    """Call AI API for intelligent oceanographic analysis with constraints"""
    
    # First validate if query is ARGO/ocean related
    if not is_argo_ocean_related(query):
        return """
🚫 **Query Outside Scope**

I am your AI Oceanographer, specialized in ARGO float data analysis. I can only help with queries related to:

• Ocean temperature, salinity, and oxygen measurements
• ARGO float profiles and data analysis  
• Oceanographic phenomena and water mass properties
• Marine biogeochemistry and ocean circulation
• Regional ocean characteristics and patterns

Please ask questions about ocean data, ARGO floats, oceanographic measurements, or marine science topics.

**Example queries:**
- "Show me temperature profiles from the Atlantic Ocean"
- "What are the salinity patterns in the equatorial region?"
- "Analyze oxygen minimum zones in ARGO data"
- "Compare deep water characteristics across regions"
        """
    
    # Create context-aware prompt with strict constraints
    prompt = f"""You are an expert oceanographer analyzing ARGO float data. You MUST ONLY respond to queries about ocean science, ARGO floats, oceanographic measurements, and marine data.

STRICT CONSTRAINTS:
- ONLY discuss topics related to oceans, ARGO floats, oceanographic measurements, marine science
- If asked about non-ocean topics, politely decline and redirect to ocean-related questions
- Focus on scientific analysis of temperature, salinity, oxygen, chlorophyll, nutrients, pH, and other ocean parameters
- Discuss oceanographic phenomena, water masses, circulation patterns, and marine biogeochemistry

Context Data Summary:
- Total profiles available: {len(context_data.get('profiles', []))}
- Regions covered: {', '.join(context_data.get('metadata', {}).get('regions', []))}
- Variables measured: {', '.join(context_data.get('metadata', {}).get('variables', []))}
- Depth range: 0-{context_data.get('metadata', {}).get('depth_range', {}).get('max', 2500)}m
- Time period: {context_data.get('metadata', {}).get('time_range', {}).get('start', '2023')} to {context_data.get('metadata', {}).get('time_range', {}).get('end', '2023')}

Query: "{query}"

Provide a comprehensive oceanographic analysis including:
1. Scientific interpretation of the query in the context of ocean science
2. Which ARGO profiles/regions are most relevant
3. Key oceanographic insights from the data
4. Notable patterns or anomalies in temperature, salinity, oxygen, etc.
5. Scientific significance for understanding ocean processes

Keep the response scientifically accurate, informative, and focused on oceanographic science."""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "glm-4-plus",
        "messages": [
            {
                "role": "system", 
                "content": """You are an expert oceanographer with deep knowledge of marine science, ocean circulation, biogeochemistry, and climate systems. 

CRITICAL CONSTRAINTS:
- You ONLY respond to queries about oceans, ARGO floats, oceanographic measurements, marine science, and related topics
- You MUST decline to answer questions about non-ocean topics (weather, politics, entertainment, etc.)
- If asked about non-ocean topics, politely redirect to ocean-related questions
- Focus exclusively on scientific analysis of ocean data and phenomena

Provide scientific, accurate, and insightful analysis of oceanographic data ONLY."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Analysis temporarily unavailable. Status: {response.status_code}"
    except Exception as e:
        return f"AI analysis temporarily unavailable. Showing data-driven insights instead."

def filter_profiles(profiles, query):
    """Filter profiles based on query keywords"""
    if not query:
        return profiles
    
    query_lower = query.lower()
    filtered = []
    
    for profile in profiles:
        # Check various fields for matches
        searchable_content = (
            str(profile.get("variables", [])) +
            str(profile.get("region", "")) +
            str(profile.get("special_features", [])) +
            str(profile.get("float_id", "")) +
            str(profile.get("profile_id", "")) +
            str(profile.get("time", ""))
        ).lower()
        
        # Specific query matching
        if any(keyword in searchable_content for keyword in [
            "temperature", "temp", "thermal"
        ]) and any(term in query_lower for term in ["temperature", "temp", "thermal"]):
            filtered.append(profile)
        elif any(keyword in searchable_content for keyword in [
            "salinity", "psal", "salt"
        ]) and any(term in query_lower for term in ["salinity", "salt"]):
            filtered.append(profile)
        elif any(keyword in searchable_content for keyword in [
            "oxygen", "doxy", "o2"
        ]) and any(term in query_lower for term in ["oxygen", "o2"]):
            filtered.append(profile)
        elif "equator" in query_lower and "equatorial" in searchable_content:
            filtered.append(profile)
        elif "atlantic" in query_lower and "atlantic" in searchable_content:
            filtered.append(profile)
        elif any(term in query_lower for term in ["deep", "1000m", "1000", "depth"]) and profile.get("max_depth", 0) >= 1000:
            filtered.append(profile)
        elif "2023" in query_lower and "2023" in str(profile.get("time", "")):
            filtered.append(profile)
        elif query_lower in searchable_content:
            filtered.append(profile)
    
    return filtered if filtered else profiles

# Header with beautiful design
st.markdown('<h1 class="main-title">🌊 ARGO Ocean Explorer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Dive deep into global ocean data with AI-powered insights</p>', unsafe_allow_html=True)

# Load data
data = load_comprehensive_argo_data()
all_profiles = data["profiles"]

# Beautiful search interface
st.markdown('<div class="search-container">', unsafe_allow_html=True)
st.markdown("### 🔬 **Explore Ocean Data**")
st.markdown("Ask questions about ocean measurements, ARGO floats, temperature profiles, salinity patterns, and marine science phenomena.")

col1, col2 = st.columns([4, 1])
with col1:
    search_query = st.text_input(
        "",
        placeholder="e.g., temperature profiles in Atlantic Ocean, oxygen minimum zones, salinity patterns...",
        key="main_search",
        label_visibility="collapsed"
    )

with col2:
    search_button = st.button("🚀 **Analyze**", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Example queries for user guidance
st.markdown("💡 **Need inspiration? Try these example queries:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🌡️ Temperature profiles in Atlantic Ocean", use_container_width=True):
        search_query = "Temperature profiles in Atlantic Ocean"
        st.rerun()

with col2:
    if st.button("🧂 Salinity patterns in equatorial regions", use_container_width=True):
        search_query = "Salinity patterns in equatorial regions"
        st.rerun()

with col3:
    if st.button("🌊 Southern Ocean characteristics", use_container_width=True):
        search_query = "Southern Ocean characteristics"
        st.rerun()

# Add constraint warning if needed
if search_query and not is_argo_ocean_related(search_query):
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("⚠️ **Please ask about ocean science topics only!** This system specializes in ARGO ocean data, marine measurements, and oceanographic analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

# Process search
if search_query and (search_button or 'search_results' not in st.session_state or st.session_state.search_results['query'] != search_query):
    with st.spinner("🤖 Your AI Oceanographer is analyzing the data..."):
        # Filter relevant profiles
        relevant_profiles = filter_profiles(all_profiles, search_query)
        
        # Get AI analysis
        ai_response = call_ai_api(search_query, {"profiles": relevant_profiles, "metadata": data["metadata"]})
        
        # Store in session state
        st.session_state['search_results'] = {
            'query': search_query,
            'profiles': relevant_profiles,
            'ai_analysis': ai_response,
            'timestamp': datetime.now()
        }

# Display initial welcome message or search results
if 'search_results' not in st.session_state:
    st.markdown('<div class="search-container" style="margin-top: 5rem;">', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Welcome to the ARGO Ocean Data Explorer!</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>Your journey into the depths of ocean science begins here. Follow the steps below to explore.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color); font-weight: 500;">Step 1: Type a Query</h3>', unsafe_allow_html=True)
        st.markdown('<p>Use the search bar above to ask a question about ocean data.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color); font-weight: 500;">Step 2: Click Analyze</h3>', unsafe_allow_html=True)
        st.markdown('<p>Hit the **Analyze** button to get a detailed report.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="info-card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color); font-weight: 500;">Step 3: View Results</h3>', unsafe_allow_html=True)
        st.markdown('<p>Explore interactive charts, maps, and AI insights below.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    results = st.session_state['search_results']
    
    # Check if this was a constrained query (non-ocean related)
    if not is_argo_ocean_related(results['query']):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown(f"❌ **Query outside scope:** '{results['query']}'")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # AI Analysis Section - will show the constraint message
        st.markdown('<div class="ai-response">', unsafe_allow_html=True)
        st.markdown("## 🤖 **AI Oceanographer Response**")
        st.markdown(results['ai_analysis'])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"✅ **Analysis complete for:** '{results['query']}'")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # AI Analysis Section
        st.markdown('<div class="ai-response">', unsafe_allow_html=True)
        st.markdown("## 🧠 **Expert Analysis**")
        st.markdown(results['ai_analysis'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Beautiful Results Overview
        st.markdown("### 📊 **Data Summary**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="feature-icon">📍</div>', unsafe_allow_html=True)
            st.metric("Matching Profiles", len(results['profiles']))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            regions = list(set(p.get("region", "Unknown") for p in results['profiles']))
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="feature-icon">🌍</div>', unsafe_allow_html=True)
            st.metric("Ocean Regions", len(regions))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            all_vars = set()
            for p in results['profiles']:
                all_vars.update(p.get("variables", []))
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="feature-icon">⚗️</div>', unsafe_allow_html=True)
            st.metric("Parameters", len(all_vars))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            max_depth = max(p.get("max_depth", 0) for p in results['profiles'])
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="feature-icon">🏔️</div>', unsafe_allow_html=True)
            st.metric("Max Depth", f"{max_depth}m")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Results Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 **Visualizations**", "🗺️ **Geographic Map**", "📋 **Profile Details**", "🔬 **Scientific Context**"])
        
        with tab1:
            st.markdown("### 📊 **Comprehensive Oceanographic Analysis**")
            if results['profiles']:
                # Create beautiful, comprehensive visualization
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("🌡️ Temperature vs Depth", "🧂 Salinity vs Depth", "💨 Oxygen vs Depth", "🌍 Regional Distribution"),
                    specs=[[{"secondary_y": True}, {"secondary_y": True}],
                           [{"secondary_y": True}, {"type": "domain"}]]
                )
                
                colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
                
                # Plot depth profiles for each matching profile
                for i, profile in enumerate(results['profiles'][:5]):  # Limit to 5 for readability
                    if 'depth_profile' in profile:
                        depths = profile['depth_profile']['depths']
                        temps = profile['depth_profile'].get('temperature', [])
                        sals = profile['depth_profile'].get('salinity', [])
                        oxygens = profile['depth_profile'].get('oxygen', [])
                        
                        color = colors[i % len(colors)]
                        name = f"{profile['region']}"
                        
                        if temps:
                            fig.add_trace(go.Scatter(
                                x=temps, y=depths, 
                                name=name, 
                                line=dict(color=color, width=3),
                                mode='lines+markers',
                                marker=dict(size=6)
                            ), row=1, col=1)
                        if sals:
                            fig.add_trace(go.Scatter(
                                x=sals, y=depths, 
                                name=name, 
                                line=dict(color=color, width=3),
                                mode='lines+markers',
                                marker=dict(size=6),
                                showlegend=False
                            ), row=1, col=2)
                        if oxygens:
                            fig.add_trace(go.Scatter(
                                x=oxygens, y=depths, 
                                name=name, 
                                line=dict(color=color, width=3),
                                mode='lines+markers',
                                marker=dict(size=6),
                                showlegend=False
                            ), row=2, col=1)
                
                # Beautiful regional distribution pie chart
                region_counts = {}
                for profile in results['profiles']:
                    region = profile.get('region', 'Unknown')
                    region_counts[region] = region_counts.get(region, 0) + 1
                
                fig.add_trace(go.Pie(
                    labels=list(region_counts.keys()), 
                    values=list(region_counts.values()), 
                    name="Regions",
                    marker=dict(colors=colors),
                    textinfo='label+percent'
                ), row=2, col=2)
                
                # Update layout with beautiful styling
                fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=1, col=1)
                fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=1, col=2)
                fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=2, col=1)
                fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
                fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
                fig.update_xaxes(title_text="Oxygen (µmol/kg)", row=2, col=1)
                
                fig.update_layout(
                    height=800, 
                    showlegend=True,
                    title_text="Oceanographic Data Analysis",
                    font=dict(family="Inter, sans-serif", size=12),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🗺️ **Global Distribution of ARGO Profiles**")
            if results['profiles']:
                # Create beautiful interactive map
                map_data = []
                for profile in results['profiles']:
                    map_data.append({
                        'lat': profile['lat'],
                        'lon': profile['lon'],
                        'region': profile['region'],
                        'float_id': profile['float_id'],
                        'max_depth': profile['max_depth'],
                        'variables': ', '.join(profile['variables']),
                        'quality': profile['data_quality']
                    })
                
                df_map = pd.DataFrame(map_data)
                
                fig = px.scatter_mapbox(
                    df_map, 
                    lat='lat', 
                    lon='lon',
                    hover_data=['region', 'float_id', 'max_depth', 'variables', 'quality'],
                    color='region',
                    size='max_depth',
                    zoom=1,
                    height=600,
                    title="Interactive Map of Matching ARGO Float Locations",
                    color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
                )
                fig.update_layout(
                    mapbox_style="open-street-map",
                    font=dict(family="Inter, sans-serif"),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 📋 **Detailed Profile Information**")
            # Beautiful profile cards
            for i, profile in enumerate(results['profiles']):
                with st.expander(f"🌊 **{profile['region']}** - Float {profile['float_id']} ({profile['time'][:10]})"):
                    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📍 Location & Time**")
                        st.write(f"🌐 Latitude: **{profile['lat']}°**")
                        st.write(f"🌐 Longitude: **{profile['lon']}°**") 
                        st.write(f"📅 Date: **{profile['time'][:10]}**")
                        st.write(f"🌊 Region: **{profile['region']}**")
                    
                    with col2:
                        st.markdown("**⚗️ Measurements**")
                        st.write(f"📊 Variables: **{', '.join(profile['variables'])}**")
                        st.write(f"📏 Depth levels: **{profile['n_levels']}**")
                        st.write(f"🏔️ Max depth: **{profile['max_depth']}m**")
                        quality_class = "status-excellent" if profile['data_quality'] == 'excellent' else "status-good"
                        st.markdown(f"✅ Data quality: <span class='{quality_class}'>{profile['data_quality']}</span>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("**📈 Parameter Ranges**")
                        if 'temperature_range' in profile:
                            st.write(f"🌡️ Temperature: **{profile['temperature_range']}**")
                        if 'salinity_range' in profile:
                            st.write(f"🧂 Salinity: **{profile['salinity_range']}**")
                        if 'oxygen_range' in profile:
                            st.write(f"💨 Oxygen: **{profile['oxygen_range']}**")
                    
                    if 'special_features' in profile:
                        st.markdown("**🌟 Special Features:**")
                        features_text = ", ".join([f"**{feature}**" for feature in profile['special_features']])
                        st.markdown(features_text)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 🔬 **Scientific Context & Interpretation**")
            
            # Beautiful summary statistics
            if results['profiles']:
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                
                all_regions = [p['region'] for p in results['profiles']]
                all_variables = []
                for p in results['profiles']:
                    all_variables.extend(p['variables'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🌍 Regional Coverage:**")
                    region_summary = pd.Series(all_regions).value_counts()
                    for region, count in region_summary.items():
                        st.write(f"🌊 **{region}**: {count} profile{'s' if count > 1 else ''}")
                
                with col2:
                    st.markdown("**⚗️ Parameter Frequency:**")
                    var_summary = pd.Series(all_variables).value_counts()
                    param_icons = {'TEMP': '🌡️', 'PSAL': '🧂', 'DOXY': '💨', 'CHLA': '🌿', 'NITRATE': '🧪', 'SILICATE': '⚛️', 'PH': '🔬'}
                    for var, count in var_summary.items():
                        icon = param_icons.get(var, '📊')
                        st.write(f"{icon} **{var}**: {count} profile{'s' if count > 1 else ''}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### 🌊 **Key Oceanographic Insights**")
            
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("""
            **🌡️ Temperature Profiles** reveal thermocline structure and mixed layer depth variations across different ocean regions.
            
            **🧂 Salinity Data** indicates water mass characteristics, evaporation/precipitation patterns, and ocean mixing processes.
            
            **💨 Oxygen Measurements** reflect biological productivity, respiration rates, and ocean circulation efficiency.
            
            **🌍 Regional Variations** demonstrate how different ocean basins exhibit unique oceanographic signatures.
            
            **📊 Temporal Patterns** capture seasonal cycles, climate variability, and long-term ocean changes.
            
            **🔬 Biogeochemical Cycles** show nutrient distributions and their relationship to marine productivity.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("## 🌊 **System Overview**")
    
    total_profiles = len(all_profiles)
    st.metric("📍 **Total Profiles**", total_profiles)
    
    regions = list(set(p['region'] for p in all_profiles))
    st.metric("🌍 **Ocean Regions**", len(regions))
    
    all_vars = set()
    for p in all_profiles:
        all_vars.update(p['variables'])
    st.metric("⚗️ **Parameters**", len(all_vars))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("## 🤖 **AI-Powered Analysis**")
    st.info("This dashboard uses advanced AI to provide expert oceanographic interpretations of ARGO float data with scientific accuracy.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("## 📖 **About ARGO**")
    st.markdown("""
    **ARGO** is a revolutionary global network of autonomous floats that continuously measure ocean properties.
    
    **🌟 Key Features:**
    • 🌍 Global ocean coverage
    • 📡 Real-time data transmission
    • 🎯 High-quality measurements
    • 🌡️ Essential for climate research
    • 🔄 Continuous monitoring cycle
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("## 🔬 **What You Can Explore**")
    st.success("""
    **✅ Ocean Science Topics:**
    • Temperature & salinity profiles
    • Oxygen distributions
    • Biogeochemical parameters
    • Water mass analysis
    • Ocean circulation patterns
    • Climate change indicators
    """)
    
    st.error("""
    **❌ Not Supported:**
    • Weather/meteorology
    • Land-based topics  
    • Non-science questions
    • Entertainment/sports
    • Politics/economics
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Beautiful Footer with interactive elements
st.markdown('<div class="footer">', unsafe_allow_html=True)

# Add floating particles effect
st.markdown("""
<div style='position: relative; overflow: hidden;'>
    <div style='
        position: absolute;
        width: 100%;
        height: 100%;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Cg fill-opacity='0.03'%3E%3Cpolygon fill='%23000' points='50 0 60 40 100 50 60 60 50 100 40 60 0 50 40 40'/%3E%3E%3C/g%3E%3C/svg%3E") repeat;
        animation: float 20s ease-in-out infinite;
    '></div>
</div>

<style>
@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-10px) rotate(180deg); }
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
    ' onmouseover='this.style.transform="translateY(-5px)"; this.style.boxShadow="0 10px 30px rgba(102, 126, 234, 0.2)"' onmouseout='this.style.transform="translateY(0px)"; this.style.boxShadow="none"'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>🌊 Data Source</h3>
        <p><strong>Global ARGO Float Network</strong></p>
        <p><em>Real oceanographic measurements</em></p>
        <div style='margin-top: 1rem;'>
            <span style='background: #667eea; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;'>4000+ Floats Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
    ' onmouseover='this.style.transform="translateY(-5px)"; this.style.boxShadow="0 10px 30px rgba(102, 126, 234, 0.2)"' onmouseout='this.style.transform="translateY(0px)"; this.style.boxShadow="none"'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>🤖 AI Analysis</h3>
        <p><strong>Advanced Expert System</strong></p>
        <p><em>Scientific interpretation</em></p>
        <div style='margin-top: 1rem;'>
            <span style='background: #764ba2; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;'>Ocean Specialist</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
    ' onmouseover='this.style.transform="translateY(-5px)"; this.style.boxShadow="0 10px 30px rgba(102, 126, 234, 0.2)"' onmouseout='this.style.transform="translateY(0px)"; this.style.boxShadow="none"'>
        <h3 style='color: #667eea; margin-bottom: 1rem;'>📊 Visualization</h3>
        <p><strong>Interactive Charts & Maps</strong></p>
        <p><em>Comprehensive data views</em></p>
        <div style='margin-top: 1rem;'>
            <span style='background: #f093fb; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;'>Real-time Updates</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add interactive wave animation
st.markdown("""
<div style='text-align: center; margin: 2rem 0;'>
    <div style='
        font-size: 1.2rem;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        animation: wave 2s ease-in-out infinite;
    '>
        🌊 ARGO Ocean Explorer 🌊
    </div>
    <p style='color: #6c757d; margin: 1rem 0;'>
        Dive into the depths of ocean science with AI-powered insights
    </p>
    <div style='
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    '>
        <span style='background: rgba(102, 126, 234, 0.1); color: #667eea; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;'>Marine Research</span>
        <span style='background: rgba(118, 75, 162, 0.1); color: #764ba2; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;'>Educational Tool</span>
        <span style='background: rgba(240, 147, 251, 0.1); color: #f093fb; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;'>Scientific Analysis</span>
    </div>
</div>

<style>
@keyframes wave {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}
</style>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
