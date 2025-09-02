# dashboard.py
"""
Streamlit app for visualizing ARGO profiles with RAG chat interface.
Windows-compatible version with better error handling.
"""
import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from config import PARQUET_DIR

# Configuration
API_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="ARGO RAG Dashboard PoC",
    page_icon="🌊"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌊 ARGO RAG Dashboard (PoC)</h1>', unsafe_allow_html=True)

# Sidebar for queries and controls
with st.sidebar:
    st.header("🔍 Query Interface")
    
    # Query input
    query = st.text_input(
        "Ask a question about ARGO data:", 
        placeholder="e.g., 'salinity near equator March 2023'"
    )
    
    # Parameters
    k = st.slider("Number of results (k)", 1, 20, 5)
    
    # Query button
    ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    
    st.divider()
    
    # System status
    st.header("📊 System Status")
    
    # Check API health
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=5)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success("✅ RAG Server Online")
            
            # Show detailed status
            with st.expander("View Details"):
                st.write(f"**Database:** {health_data.get('database', 'unknown')}")
                st.write(f"**Embeddings:** {health_data.get('embeddings', 'unknown')}")
                st.write(f"**OpenAI:** {health_data.get('openai', 'unknown')}")
        else:
            st.error("❌ RAG Server Error")
    except requests.exceptions.RequestException:
        st.error("❌ RAG Server Offline")
        st.info("Make sure to run: `uvicorn rag_server:app --reload --port 8000`")
    
    # Try to get stats
    try:
        stats_resp = requests.get(f"{API_URL}/stats", timeout=5)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            st.metric("Total Profiles", stats.get("total_profiles", 0))
            
            date_range = stats.get("date_range", {})
            if date_range.get("earliest") and date_range.get("latest"):
                st.write(f"**Date Range:** {date_range['earliest'][:10]} to {date_range['latest'][:10]}")
    except:
        pass
    
    st.divider()
    
    # Usage tips
    st.header("💡 Usage Tips")
    st.info("""
    **Example queries:**
    - "temperature profiles near equator"
    - "salinity data from 2023"  
    - "profiles with oxygen measurements"
    - "floats in Atlantic Ocean"
    - "deep profiles below 1000m"
    """)

# Main content area
if ask_button and query:
    with st.spinner("🔍 Querying RAG backend..."):
        try:
            response = requests.post(
                f"{API_URL}/query", 
                json={"query": query, "k": k}, 
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            st.session_state["last_query"] = {
                "query": query, 
                "resp": response.json()
            }
            st.success("✅ Query completed successfully!")
            
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Try reducing the number of results or check if the server is responsive.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to RAG server. Make sure it's running on port 8000.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request failed: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Display results if available
if "last_query" in st.session_state:
    res = st.session_state["last_query"]["resp"]
    last_query_text = st.session_state["last_query"]["query"]
    
    # Show the query
    st.subheader(f"📝 Results for: '{last_query_text}'")
    
    # LLM Answer section
    st.subheader("🤖 AI Assistant Response")
    answer = res.get("answer")
    if answer and answer.strip() and answer != "None":
        st.write(answer)
    else:
        st.info("💡 No AI-generated answer available. Configure OpenAI API key for enhanced responses.")
    
    # Retrieved profiles section
    st.subheader("📋 Retrieved Profiles")
    results = res.get("results", [])
    
    if results:
        # Create DataFrame with better formatting
        df_data = []
        for r in results:
            metadata = r.get("metadata", {})
            df_data.append({
                "Profile ID": r["profile_id"],
                "Float ID": metadata.get("float_id", "N/A"),
                "Latitude": metadata.get("lat"),
                "Longitude": metadata.get("lon"),
                "Time": metadata.get("time", "N/A"),
                "Similarity": f"{r.get('similarity_score', 0):.3f}" if r.get('similarity_score') else "N/A",
                "Parquet Path": metadata.get("parquet", "N/A")
            })
        
        df = pd.DataFrame(df_data)
        
        # Display with formatting
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
            
            # Create map layer
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df.rename(columns={"Latitude": "lat", "Longitude": "lon"}),
                get_position=["lon", "lat"],
                get_fill_color=[255, 100, 100, 200],
                get_radius=50000,
                pickable=True,
                filled=True
            )
            
            # Set view state
            view = pdk.ViewState(
                latitude=map_df["Latitude"].mean(),
                longitude=map_df["Longitude"].mean(),
                zoom=2,
                pitch=0
            )
            
            # Display map
            st.pydeck_chart(pdk.Deck(
                layers=[layer], 
                initial_view_state=view,
                tooltip={"text": "Lat: {lat}\nLon: {lon}"}
            ))

        # Profile plotting section
        st.subheader("📈 Profile Visualization")
        
        # Profile selection
        profile_options = [""] + df["Profile ID"].tolist()
        selected_profile = st.selectbox(
            "Select a profile to plot:", 
            options=profile_options,
            help="Choose a profile to view temperature and salinity vs depth"
        )
        
        if selected_profile:
            # Find the selected profile data
            profile_row = df[df["Profile ID"] == selected_profile].iloc[0]
            parquet_path = profile_row["Parquet Path"]
            
            if parquet_path and parquet_path != "N/A" and Path(parquet_path).exists():
                try:
                    # Load parquet data
                    dfp = pd.read_parquet(parquet_path)
                    
                    # Create subplot for temperature and salinity
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Temperature Profile", "Salinity Profile"),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Temperature plot
                    if "TEMP" in dfp.columns and "PRES" in dfp.columns:
                        temp_data = dfp[["TEMP", "PRES"]].dropna()
                        if not temp_data.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=temp_data["TEMP"], 
                                    y=temp_data["PRES"],
                                    mode="lines+markers",
                                    name="Temperature",
                                    line=dict(color="red", width=2),
                                    marker=dict(size=4)
                                ),
                                row=1, col=1
                            )
                    
                    # Salinity plot
                    if "PSAL" in dfp.columns and "PRES" in dfp.columns:
                        sal_data = dfp[["PSAL", "PRES"]].dropna()
                        if not sal_data.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=sal_data["PSAL"], 
                                    y=sal_data["PRES"],
                                    mode="lines+markers",
                                    name="Salinity",
                                    line=dict(color="blue", width=2),
                                    marker=dict(size=4)
                                ),
                                row=1, col=2
                            )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Profile Data: {selected_profile}",
                        height=600,
                        showlegend=False
                    )
                    
                    # Update axes
                    fig.update_yaxes(title_text="Pressure (dbar)", autorange="reversed", row=1, col=1)
                    fig.update_yaxes(title_text="Pressure (dbar)", autorange="reversed", row=1, col=2)
                    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
                    fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data preview
                    with st.expander("📊 View Raw Data"):
                        st.write(f"**Shape:** {dfp.shape[0]} rows × {dfp.shape[1]} columns")
                        st.write(f"**Columns:** {', '.join(dfp.columns)}")
                        st.dataframe(dfp.head(20))
                        
                        # Download button for data
                        csv_data = dfp.to_csv(index=False)
                        st.download_button(
                            label="💾 Download as CSV",
                            data=csv_data,
                            file_name=f"{selected_profile.replace(':', '_')}.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error loading profile data: {str(e)}")
                    st.info("The parquet file might be corrupted or in an incompatible format.")
            else:
                st.warning("📁 Parquet file not found. The file may have been moved or the ingestion path differs.")
                st.info(f"Expected path: `{parquet_path}`")
    else:
        st.info("🔍 No profiles found for your query. Try a different search term or check if data has been ingested.")

# Footer with information
st.divider()

# Two columns for footer info
col1, col2 = st.columns(2)

with col1:
    st.subheader("ℹ️ About This PoC")
    st.write("""
    This dashboard provides a RAG (Retrieval-Augmented Generation) interface for ARGO oceanographic data.
    
    **Features:**
    - Natural language queries over ARGO float profiles
    - Geographic visualization of matching profiles  
    - Interactive depth profile plotting
    - AI-powered responses (when OpenAI is configured)
    """)

with col2:
    st.subheader("🚀 Quick Start")
    st.write("""
    **To get started:**
    1. Ensure PostgreSQL and RAG server are running
    2. Ingest some ARGO NetCDF data using `ingest.py`
    3. Enter a natural language query in the sidebar
    4. Explore the results on the map and plots
    
    **Need data?** Check the README for links to ARGO data sources.
    """)

# Advanced features in expander
with st.expander("🔧 Advanced Features"):
    st.write("**Development Status:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗄️ View Database Stats"):
            try:
                stats_resp = requests.get(f"{API_URL}/stats", timeout=10)
                if stats_resp.status_code == 200:
                    stats = stats_resp.json()
                    st.json(stats)
                else:
                    st.error("Failed to get stats")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("🔄 Clear Cache"):
            st.cache_data.clear()
            if "last_query" in st.session_state:
                del st.session_state["last_query"]
            st.success("Cache cleared!")
    
    with col3:
        st.write("**API Endpoints:**")
        st.code(f"{API_URL}/docs")

# Debug information (only show if there's an error)
if st.checkbox("🐛 Show Debug Info"):
    st.subheader("Debug Information")
    
    # Show session state
    st.write("**Session State:**")
    st.json(dict(st.session_state))
    
    # Show configuration
    st.write("**Configuration:**")
    st.code(f"""
    API_URL: {API_URL}
    PARQUET_DIR: {PARQUET_DIR}
    """)

# Help section
with st.expander("❓ Help & Troubleshooting"):
    st.write("""
    **Common Issues:**
    
    1. **"RAG Server Offline"**
       - Make sure you've started the server: `uvicorn rag_server:app --reload --port 8000`
       - Check if port 8000 is available
    
    2. **"No profiles found"**
       - Ensure you've run the ingestion script: `python ingest.py --dir /path/to/netcdf`
       - Check if PostgreSQL is running: `docker-compose ps`
    
    3. **"Parquet file not found"**
       - Verify the parquet files exist in the configured directory
       - Check file permissions
    
    4. **Slow performance**
       - Reduce the number of results (k parameter)
       - Ensure enough RAM is available for the embedding model
    
    **For more help, check the README.md file in your project directory.**
    """)

# Footer
st.markdown("---")
st.markdown(
    "🌊 **ARGO RAG PoC** | Built with Streamlit, FastAPI, and ChromaDB | "
    "For production use, consider scaling tips in the documentation."
)
