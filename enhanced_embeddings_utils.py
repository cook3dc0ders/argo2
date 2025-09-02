# enhanced_embeddings_utils.py
"""
Enhanced embedding utilities optimized for large ARGO datasets (50-60GB).
Includes batch processing, memory management, and improved text generation.
"""
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import json
import numpy as np
from datetime import datetime
import time
from enhanced_config import (
    CHROMA_DIR, EMBEDDING_MODEL, CHROMA_BATCH_SIZE, 
    MAX_PROFILES_IN_MEMORY, CHECKPOINT_INTERVAL,
    GEOGRAPHIC_REGIONS, PARAMETER_GROUPS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEmbeddingManager:
    """Enhanced embedding manager for large-scale ARGO data processing"""
    
    def __init__(self, model_name=EMBEDDING_MODEL, persist_dir=CHROMA_DIR):
        self.model_name = model_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.embeddings_created = 0
        self.batch_count = 0
        
        try:
            # Initialize sentence transformer with optimization for scientific text
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            
            # Optimize for batch processing
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = 384  # Reasonable limit for ARGO descriptions
            
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise
        
        try:
            # Initialize ChromaDB with optimized settings
            logger.info(f"Initializing ChromaDB: {self.persist_dir}")
            
            # Use PersistentClient with optimized settings
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            self._initialize_collection()
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _initialize_collection(self):
        """Initialize ChromaDB collection with optimized settings"""
        collection_name = "argo_profiles_enhanced"
        
        try:
            # Try to get existing collection
            self.col = self.client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
            
        except Exception:
            # Create new collection with enhanced embedding function
            try:
                logger.info(f"Creating new collection: {collection_name}")
                
                # Create embedding function
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.model_name
                )
                
                self.col = self.client.create_collection(
                    name=collection_name, 
                    embedding_function=ef,
                    metadata={"description": "Enhanced ARGO profiles with improved summaries"}
                )
                logger.info("New collection created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                raise

    def create_enhanced_profile_summary(self, profile_data: Dict) -> str:
        """Create an enhanced summary optimized for oceanographic queries"""
        try:
            summary_parts = []
            
            # Extract key information
            float_id = profile_data.get('float_id', 'unknown')
            profile_id = profile_data.get('profile_id', 'unknown')
            lat = profile_data.get('lat')
            lon = profile_data.get('lon')
            time_str = profile_data.get('time')
            variables = profile_data.get('variables', [])
            n_levels = profile_data.get('n_levels', 0)
            
            # 1. Basic identification with context
            summary_parts.append(f"ARGO oceanographic profile {profile_id} from float {float_id}")
            
            # 2. Enhanced geographic description
            if lat is not None and lon is not None:
                region_name = self._identify_ocean_region(lat, lon)
                depth_zone = self._get_depth_zone_description(n_levels)
                
                # Add specific coordinate and region
                summary_parts.append(f"located at {lat:.3f}°N {lon:.3f}°E in the {region_name}")
                
                # Add geographic context keywords for better matching
                if abs(lat) <= 5:
                    summary_parts.append("near equatorial waters")
                elif 'arabian' in region_name.lower():
                    summary_parts.append("in Arabian Sea waters")
                elif 'pacific' in region_name.lower():
                    summary_parts.append("in Pacific Ocean basin")
                elif 'atlantic' in region_name.lower():
                    summary_parts.append("in Atlantic Ocean basin")
                
                summary_parts.append(depth_zone)
            
            # 3. Enhanced temporal description
            if time_str:
                try:
                    if isinstance(time_str, str):
                        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    else:
                        dt = time_str
                    
                    # Add multiple temporal contexts for better matching
                    month_name = dt.strftime('%B')
                    year = dt.year
                    season = self._get_season(dt.month)
                    
                    summary_parts.append(f"measured in {month_name} {year}")
                    summary_parts.append(f"during {season} season")
                    
                    # Add relative time context
                    days_ago = (datetime.now() - dt).days
                    if days_ago < 180:
                        summary_parts.append("recent measurements")
                    elif days_ago < 365:
                        summary_parts.append("within last year")
                    
                except Exception as e:
                    logger.debug(f"Time parsing error: {e}")
                    summary_parts.append(f"measured at {time_str}")
            
            # 4. Enhanced parameter description
            if variables:
                # Categorize parameters
                core_params = [v for v in variables if v.upper() in PARAMETER_GROUPS['core']]
                bgc_params = [v for v in variables if v.upper() in PARAMETER_GROUPS['bgc']]
                
                if bgc_params:
                    summary_parts.append(f"with biogeochemical parameters: {', '.join(bgc_params)}")
                    summary_parts.append("BGC-Argo float data")
                    
                    # Add specific BGC context
                    if any('DOXY' in v.upper() for v in bgc_params):
                        summary_parts.append("including dissolved oxygen measurements")
                    if any('NITRATE' in v.upper() for v in bgc_params):
                        summary_parts.append("including nitrate nutrient data")
                    if any('CHLA' in v.upper() for v in bgc_params):
                        summary_parts.append("including chlorophyll fluorescence")
                
                if core_params:
                    summary_parts.append(f"with core measurements: {', '.join(core_params)}")
                    
                    # Add specific parameter context
                    if any('TEMP' in v.upper() for v in variables):
                        summary_parts.append("temperature profile data")
                    if any('PSAL' in v.upper() for v in variables):
                        summary_parts.append("salinity profile measurements")
                    if any('PRES' in v.upper() for v in variables):
                        summary_parts.append("pressure depth measurements")
            
            # 5. Depth and profile characteristics
            if n_levels > 0:
                if n_levels > 150:
                   