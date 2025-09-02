#!/usr/bin/env python3
"""
Ultra-robust ARGO NetCDF ingestion system.
Automatically adapts to ANY NetCDF file structure by:
1. Dynamically discovering all variables and dimensions
2. Creating database columns on-the-fly as needed
3. Handling any data type and structure
4. Never failing due to schema mismatches
"""
import argparse
import os
from pathlib import Path
import json
import pandas as pd
import xarray as xr
from datetime import datetime
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, JSON, inspect
from sqlalchemy.dialects.postgresql import JSONB
from tqdm import tqdm
from embeddings_utils import EmbeddingManager
from config import PARQUET_DIR, PG
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pg_engine():
    """Create PostgreSQL engine"""
    url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
    return create_engine(url, pool_pre_ping=True)

class DynamicArgoIngestor:
    """
    Dynamic ingestion system that adapts to any NetCDF structure.
    Creates database columns automatically as new variables are encountered.
    """
    
    def __init__(self, parquet_dir):
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        self.engine = pg_engine()
        self.known_columns = set()
        self.metadata = MetaData()
        
        # Load existing columns first
        self._load_existing_columns()
        
        # Initialize base table structure
        self._ensure_base_table()
        
        # Reload columns after table setup
        self._load_existing_columns()
        
        # Initialize embeddings
        try:
            self.emb = EmbeddingManager()
            logger.info("Embedding manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize embeddings: {e}")
            self.emb = None
    
    def _ensure_base_table(self):
        """Ensure the base floats table exists with core columns"""
        try:
            with self.engine.begin() as conn:
                # Check if table exists
                table_check = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'floats'
                    );
                """)).scalar()
                
                if not table_check:
                    # Create new table with full structure
                    conn.execute(text("""
                        CREATE TABLE floats (
                            id SERIAL PRIMARY KEY,
                            float_id TEXT,
                            profile_id TEXT UNIQUE,
                            source_file TEXT,
                            time_start TIMESTAMP,
                            time_end TIMESTAMP,
                            lat DOUBLE PRECISION,
                            lon DOUBLE PRECISION,
                            platform_number TEXT,
                            n_levels INTEGER DEFAULT 0,
                            variables JSONB DEFAULT '[]'::jsonb,
                            variables_list JSONB DEFAULT '[]'::jsonb,
                            parquet_path TEXT,
                            raw_metadata JSONB DEFAULT '{}'::jsonb,
                            created_at TIMESTAMP DEFAULT NOW(),
                            updated_at TIMESTAMP DEFAULT NOW()
                        );
                    """))
                    logger.info("Created new floats table")
                else:
                    # Add missing columns to existing table
                    logger.info("Table exists, ensuring all required columns...")
                    columns_to_add = [
                        ("variables_list", "JSONB DEFAULT '[]'::jsonb"),
                        ("raw_metadata", "JSONB DEFAULT '{}'::jsonb"),
                        ("time_end", "TIMESTAMP"),
                        ("platform_number", "TEXT"),
                        ("variables", "JSONB DEFAULT '[]'::jsonb")
                    ]
                    
                    for col_name, col_def in columns_to_add:
                        try:
                            conn.execute(text(f'ALTER TABLE floats ADD COLUMN IF NOT EXISTS {col_name} {col_def}'))
                        except Exception as e:
                            logger.debug(f"Could not add column {col_name}: {e}")
                
                # Create basic indexes (ignore errors for existing ones)
                basic_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_floats_profile_id ON floats(profile_id)",
                    "CREATE INDEX IF NOT EXISTS idx_floats_float_id ON floats(float_id)",
                    "CREATE INDEX IF NOT EXISTS idx_floats_time ON floats(time_start)",
                    "CREATE INDEX IF NOT EXISTS idx_floats_location ON floats(lat, lon)"
                ]
                
                for idx_sql in basic_indexes:
                    try:
                        conn.execute(text(idx_sql))
                    except Exception as e:
                        logger.debug(f"Index creation warning: {e}")
                
                logger.info("Base table structure ensured")
        except Exception as e:
            logger.error(f"Error creating base table: {e}")
            raise
    
    def _load_existing_columns(self):
        """Load existing column names from the database"""
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns('floats')
            self.known_columns = {col['name'] for col in columns}
            logger.info(f"Loaded {len(self.known_columns)} existing columns")
        except Exception as e:
            logger.warning(f"Could not load existing columns: {e}")
            self.known_columns = {'id', 'float_id', 'profile_id', 'source_file', 'time_start', 
                                'lat', 'lon', 'n_levels', 'variables_list', 'parquet_path', 
                                'raw_metadata', 'created_at', 'updated_at'}
    
    def _safe_column_name(self, name):
        """Convert any variable name to a safe PostgreSQL column name"""
        # Replace problematic characters
        safe_name = str(name).lower()
        safe_name = ''.join(c if c.isalnum() else '_' for c in safe_name)
        safe_name = safe_name.strip('_')
        
        # Ensure it doesn't start with a number
        if safe_name and safe_name[0].isdigit():
            safe_name = 'var_' + safe_name
        
        # Ensure it's not empty
        if not safe_name:
            safe_name = 'unknown_var'
        
        # Limit length
        if len(safe_name) > 60:
            safe_name = safe_name[:60]
        
        return safe_name
    
    def _add_column_if_needed(self, column_name, sample_value):
        """Add a new column to the table if it doesn't exist"""
        if column_name in self.known_columns:
            return True
        
        try:
            # Determine column type based on sample value and column name
            if 'time' in column_name.lower() and column_name != 'time_start':
                # For time-related columns that aren't time_start, use TEXT to avoid timestamp issues
                col_type = "TEXT"
            elif isinstance(sample_value, (int, np.integer)):
                col_type = "INTEGER"
            elif isinstance(sample_value, (float, np.floating)):
                col_type = "DOUBLE PRECISION"
            elif isinstance(sample_value, datetime):
                col_type = "TIMESTAMP"
            elif isinstance(sample_value, (list, dict)):
                col_type = "JSONB"
            else:
                col_type = "TEXT"
            
            # Add the column
            with self.engine.begin() as conn:
                sql = f'ALTER TABLE floats ADD COLUMN IF NOT EXISTS "{column_name}" {col_type}'
                conn.execute(text(sql))
                
            self.known_columns.add(column_name)
            logger.info(f"Added new column: {column_name} ({col_type})")
            return True
            
        except Exception as e:
            logger.warning(f"Could not add column {column_name}: {e}")
            return False
    
    def _safe_extract_value(self, ds, var_name, profile_idx=None):
        """Safely extract any value from a NetCDF variable"""
        try:
            if var_name not in ds.variables:
                return None
            
            var = ds[var_name]
            
            # Handle profile indexing for multi-profile files
            if profile_idx is not None and var.ndim > 0:
                # Try to index by the first dimension if it matches profile count
                if var.shape[0] > profile_idx:
                    try:
                        var = var.isel({var.dims[0]: profile_idx})
                    except:
                        pass
            
            # Get the values
            values = var.values
            
            # Handle different data types
            if values.ndim == 0:
                # Scalar value
                val = values.item()
                
                # Handle bytes/strings
                if isinstance(val, bytes):
                    return val.decode('utf-8', errors='ignore').strip('\x00 ')
                elif isinstance(val, str):
                    return val.strip('\x00 ')
                elif pd.isna(val):
                    return None
                else:
                    return val
            
            elif values.ndim == 1:
                # Array of values
                if values.dtype.kind in 'SU':  # String types
                    # Handle string arrays
                    try:
                        if hasattr(values, 'astype'):
                            str_values = values.astype(str)
                            return [s.strip('\x00 ') for s in str_values if s and s.strip('\x00 ')]
                        else:
                            return [str(v).strip('\x00 ') for v in values if v]
                    except:
                        return str(values)
                else:
                    # Numeric arrays
                    clean_values = []
                    for v in values:
                        if pd.isna(v) or np.isinf(v):
                            clean_values.append(None)
                        else:
                            clean_values.append(float(v) if values.dtype.kind in 'fc' else int(v))
                    return clean_values if any(v is not None for v in clean_values) else None
            
            else:
                # Multi-dimensional - flatten and take first valid values
                flat_values = values.flatten()
                if len(flat_values) > 0:
                    return self._safe_extract_value(ds, var_name, 0)  # Recursive call with flattened
                
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract {var_name}: {e}")
            return None
    
    def _discover_file_structure(self, ds):
        """Discover the complete structure of a NetCDF file"""
        structure = {
            'dimensions': dict(ds.dims),
            'variables': {},
            'attributes': dict(ds.attrs),
            'is_multi_profile': False,
            'profile_count': 1,
            'profile_dim': None
        }
        
        # Detect if this is a multi-profile file
        profile_dims = ['N_PROF', 'N_PROFILES', 'profile']
        for dim_name in profile_dims:
            if dim_name in ds.dims and ds.dims[dim_name] > 1:
                structure['is_multi_profile'] = True
                structure['profile_count'] = ds.dims[dim_name]
                structure['profile_dim'] = dim_name
                break
        
        # Catalog all variables
        for var_name, var in ds.variables.items():
            try:
                structure['variables'][var_name] = {
                    'dims': var.dims,
                    'shape': var.shape,
                    'dtype': str(var.dtype),
                    'attributes': dict(var.attrs) if hasattr(var, 'attrs') else {}
                }
            except:
                structure['variables'][var_name] = {'dims': [], 'shape': [], 'dtype': 'unknown'}
        
        return structure
    
    def _extract_profile_data(self, ds, structure, profile_idx=None):
        """Extract all possible data from a profile"""
        profile_data = {}
        
        # Extract all variables
        for var_name in structure['variables'].keys():
            value = self._safe_extract_value(ds, var_name, profile_idx)
            if value is not None:
                safe_name = self._safe_column_name(var_name)
                profile_data[safe_name] = value
        
        # Create a unique profile ID
        source_name = getattr(ds, 'source', 'unknown')
        if hasattr(ds, 'encoding') and hasattr(ds.encoding, 'source'):
            source_name = Path(ds.encoding.source).stem
        
        if profile_idx is not None:
            profile_id = f"{source_name}_profile_{profile_idx}"
        else:
            profile_id = f"{source_name}_profile_0"
        
        profile_data['profile_id'] = profile_id
        
        # Try to extract standard coordinates with fallbacks
        lat = self._find_coordinate(profile_data, ['latitude', 'lat', 'y'])
        lon = self._find_coordinate(profile_data, ['longitude', 'lon', 'x'])
        time_val = self._find_coordinate(profile_data, ['juld', 'time', 'date_time', 'reference_date_time'])
        
        # Ensure we have basic coordinates
        if lat is None or lon is None:
            # Try to find ANY numeric values that could be coordinates
            for key, value in profile_data.items():
                if isinstance(value, (int, float)):
                    if -90 <= value <= 90 and lat is None:
                        lat = value
                    elif -180 <= value <= 180 and lon is None:
                        lon = value
        
        profile_data['lat'] = lat
        profile_data['lon'] = lon
        
        # Handle time - convert various time formats
        if time_val is not None:
            time_str = self._convert_time_value(time_val)
            profile_data['time_start'] = time_str
        
        # Find float ID
        float_id = self._find_coordinate(profile_data, ['platform_number', 'float_id', 'wmo_inst_type'])
        if float_id is None:
            float_id = source_name
        profile_data['float_id'] = str(float_id)
        
        # Store original structure info
        profile_data['source_structure'] = structure
        profile_data['source_file'] = source_name
        
        return profile_data
    
    def _find_coordinate(self, data, possible_names):
        """Find a coordinate value by trying multiple possible column names"""
        for name in possible_names:
            # Try exact match
            if name in data:
                return data[name]
            
            # Try variations
            for key in data.keys():
                if name.lower() in key.lower() or key.lower() in name.lower():
                    return data[key]
        
        return None
    
    def _convert_time_value(self, time_val):
        """Convert various time formats to a standard timestamp string"""
        try:
            if isinstance(time_val, str):
                # Try to parse string dates
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y%m%d', '%Y']:
                    try:
                        dt = datetime.strptime(time_val, fmt)
                        return dt.isoformat()
                    except:
                        continue
                # If string parsing fails, return None for TIMESTAMP column
                return None
            
            elif isinstance(time_val, (int, float)):
                # Handle numeric time values (days since reference, unix timestamps, etc.)
                if time_val > 1e9:  # Likely unix timestamp
                    dt = datetime.fromtimestamp(time_val)
                    return dt.isoformat()
                elif time_val > 10000:  # Likely days since some epoch
                    # Assume days since 1950-01-01 (common ARGO reference)
                    reference_date = datetime(1950, 1, 1)
                    dt = reference_date + pd.Timedelta(days=time_val)
                    return dt.isoformat()
                else:
                    # Small number - can't reliably convert to timestamp
                    # Return None so it gets stored as NULL in database
                    return None
            
            elif hasattr(time_val, 'isoformat'):
                return time_val.isoformat()
            
            else:
                # Unknown time format - return None for TIMESTAMP column
                return None
                
        except Exception as e:
            logger.debug(f"Could not convert time value {time_val}: {e}")
            return None  # Return None instead of string for TIMESTAMP column
    
    def _prepare_for_database(self, profile_data):
        """Prepare profile data for database insertion"""
        db_record = {}
        level_data = {}
        
        # Separate scalar values from arrays
        for key, value in profile_data.items():
            if key in ['source_structure']:
                continue  # Skip non-database fields
            
            try:
                if isinstance(value, list) and len(value) > 1:
                    # This is level data (measurements at different depths)
                    level_data[key] = value
                elif isinstance(value, list) and len(value) == 1:
                    # Single-element list - extract the value
                    extracted_val = value[0]
                    # Special handling for time_start column
                    if key == 'time_start' and isinstance(extracted_val, (int, float)):
                        converted_time = self._convert_time_value(extracted_val)
                        db_record[key] = converted_time
                    else:
                        db_record[key] = extracted_val
                elif isinstance(value, (dict, list)):
                    # Store complex data as JSON
                    db_record[key] = json.dumps(value, default=str)
                else:
                    # Scalar value - special handling for time columns
                    if key == 'time_start' and isinstance(value, (int, float, str)):
                        converted_time = self._convert_time_value(value)
                        db_record[key] = converted_time
                    else:
                        db_record[key] = value
            except Exception as e:
                logger.debug(f"Error processing {key}: {e}")
                continue
        
        # Ensure required fields have valid values
        if 'profile_id' not in db_record or not db_record['profile_id']:
            db_record['profile_id'] = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        if 'float_id' not in db_record:
            db_record['float_id'] = 'unknown_float'
        
        # Handle coordinates
        if 'lat' in db_record:
            try:
                lat_val = float(db_record['lat'])
                if -90 <= lat_val <= 90:
                    db_record['lat'] = lat_val
                else:
                    db_record['lat'] = None
            except:
                db_record['lat'] = None
        
        if 'lon' in db_record:
            try:
                lon_val = float(db_record['lon'])
                if -180 <= lon_val <= 180:
                    db_record['lon'] = lon_val
                else:
                    db_record['lon'] = None
            except:
                db_record['lon'] = None
        
        # Set n_levels based on level data
        if level_data:
            max_levels = max(len(v) for v in level_data.values() if isinstance(v, list))
            db_record['n_levels'] = max_levels
        else:
            db_record['n_levels'] = 0
        
        # Create variables list
        variables_list = list(level_data.keys()) if level_data else []
        db_record['variables_list'] = json.dumps(variables_list)
        
        # Ensure time_start is properly handled - if still problematic, set to NULL
        if 'time_start' in db_record and isinstance(db_record['time_start'], str):
            if db_record['time_start'].startswith('relative_time_'):
                db_record['time_start'] = None  # Set to NULL instead of invalid string
        
        return db_record, level_data
    
    def _add_columns_for_record(self, db_record):
        """Add any missing columns needed for this record"""
        for column_name, value in db_record.items():
            if column_name not in self.known_columns:
                self._add_column_if_needed(column_name, value)
    
    def _write_parquet(self, profile_id, level_data, float_id='unknown'):
        """Write level data to parquet file"""
        if not level_data:
            return None
        
        try:
            # Clean IDs for file system
            clean_float_id = ''.join(c for c in str(float_id) if c.isalnum() or c in '_-')[:50]
            clean_profile_id = ''.join(c for c in str(profile_id) if c.isalnum() or c in '_-')[:100]
            
            # Create directory
            float_dir = self.parquet_dir / clean_float_id
            float_dir.mkdir(parents=True, exist_ok=True)
            
            # Create DataFrame
            df_data = {}
            max_length = 0
            
            # First pass - find maximum length
            for key, values in level_data.items():
                if isinstance(values, list):
                    max_length = max(max_length, len(values))
            
            # Second pass - pad all arrays to same length
            for key, values in level_data.items():
                if isinstance(values, list):
                    padded_values = values + [None] * (max_length - len(values))
                    df_data[key] = padded_values
                elif values is not None:
                    # Scalar value - replicate for all levels
                    df_data[key] = [values] * max_length
            
            if not df_data:
                return None
            
            # Create DataFrame and save
            df = pd.DataFrame(df_data)
            df = df.dropna(how='all')  # Remove completely empty rows
            
            if df.empty:
                return None
            
            parquet_path = float_dir / f"{clean_profile_id}.parquet"
            df.to_parquet(parquet_path, index=False)
            
            logger.debug(f"Saved parquet: {parquet_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
            return str(parquet_path)
            
        except Exception as e:
            logger.error(f"Error writing parquet for {profile_id}: {e}")
            return None
    
    def _create_embedding_summary(self, db_record, level_data):
        """Create a summary for embedding"""
        try:
            parts = []
            
            # Basic info
            float_id = db_record.get('float_id', 'unknown')
            profile_id = db_record.get('profile_id', 'unknown')
            parts.append(f"Float {float_id} profile {profile_id}")
            
            # Location
            lat = db_record.get('lat')
            lon = db_record.get('lon')
            if lat is not None and lon is not None:
                parts.append(f"at {lat:.2f}°N, {lon:.2f}°E")
            
            # Time
            time_val = db_record.get('time_start')
            if time_val:
                parts.append(f"measured on {time_val}")
            
            # Measurements
            n_levels = db_record.get('n_levels', 0)
            if n_levels > 0:
                parts.append(f"with {n_levels} depth levels")
            
            # Variables
            if level_data:
                var_names = list(level_data.keys())
                if var_names:
                    parts.append(f"measuring {', '.join(var_names[:5])}")
            
            return " ".join(parts) + "."
            
        except Exception as e:
            logger.debug(f"Error creating summary: {e}")
            return f"Profile {db_record.get('profile_id', 'unknown')}"
    
    def _insert_record(self, db_record):
        """Insert record into database with dynamic column handling"""
        try:
            # Add any missing columns
            self._add_columns_for_record(db_record)
            
            # Prepare the insert statement - only use columns that exist in the table
            available_columns = []
            values_dict = {}
            
            for key, value in db_record.items():
                safe_key = self._safe_column_name(key)
                if safe_key in self.known_columns:
                    available_columns.append(safe_key)
                    
                    # Clean the value for database insertion
                    try:
                        if isinstance(value, (np.integer, np.floating)):
                            value = value.item()
                        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                            value = None
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, default=str)
                        
                        values_dict[safe_key] = value
                    except:
                        values_dict[safe_key] = str(value) if value is not None else None
            
            # Ensure we have at least profile_id
            if 'profile_id' not in values_dict:
                logger.error("Cannot insert record without profile_id")
                return False
            
            # Build the SQL statement
            columns_str = ', '.join(f'"{col}"' for col in available_columns)
            placeholders_str = ', '.join(f':{col}' for col in available_columns)
            
            # Build update clause (all columns except profile_id and id)
            update_cols = [col for col in available_columns if col not in ['profile_id', 'id']]
            update_clause = ', '.join(f'"{col}" = EXCLUDED."{col}"' for col in update_cols)
            
            sql = f"""
                INSERT INTO floats ({columns_str})
                VALUES ({placeholders_str})
                ON CONFLICT (profile_id) DO UPDATE SET
                    updated_at = NOW()
                    {', ' + update_clause if update_clause else ''}
            """
            
            # Execute insert
            with self.engine.begin() as conn:
                conn.execute(text(sql), values_dict)
                
            return True
            
        except Exception as e:
            logger.error(f"Database insertion failed: {e}")
            logger.debug(f"Available columns: {available_columns}")
            logger.debug(f"Values dict keys: {list(values_dict.keys())}")
            return False
    
    def process_file(self, file_path):
        """Process a single NetCDF file"""
        try:
            logger.info(f"Processing {file_path}")
            
            # Open the file with multiple fallback methods
            ds = None
            for decode_times in [False, True]:
                for mask_and_scale in [True, False]:
                    try:
                        ds = xr.open_dataset(
                            file_path, 
                            decode_times=decode_times, 
                            mask_and_scale=mask_and_scale,
                            decode_coords=False  # Avoid coordinate decoding issues
                        )
                        break
                    except Exception as e:
                        logger.debug(f"Failed to open with decode_times={decode_times}, mask_and_scale={mask_and_scale}: {e}")
                        continue
                if ds is not None:
                    break
            
            if ds is None:
                logger.error(f"Could not open {file_path} with any method")
                return 0, 0
            
            # Discover file structure
            structure = self._discover_file_structure(ds)
            logger.info(f"File structure: {structure['profile_count']} profiles, "
                       f"{len(structure['variables'])} variables")
            
            successful_profiles = 0
            total_profiles = structure['profile_count']
            
            # Process each profile
            for profile_idx in range(total_profiles):
                try:
                    # Extract profile data
                    if structure['is_multi_profile']:
                        profile_data = self._extract_profile_data(ds, structure, profile_idx)
                    else:
                        profile_data = self._extract_profile_data(ds, structure, None)
                    
                    # Skip if no useful data
                    if not profile_data or len(profile_data) < 3:
                        continue
                    
                    # Prepare for database
                    db_record, level_data = self._prepare_for_database(profile_data)
                    
                    # Skip if missing critical info
                    if not db_record.get('profile_id'):
                        continue
                    
                    # Write parquet file
                    parquet_path = self._write_parquet(
                        db_record['profile_id'], 
                        level_data, 
                        db_record.get('float_id', 'unknown')
                    )
                    if parquet_path:
                        db_record['parquet_path'] = parquet_path
                    
                    # Set source file
                    db_record['source_file'] = str(file_path)
                    
                    # Insert into database
                    if self._insert_record(db_record):
                        # Create embedding
                        if self.emb:
                            try:
                                summary = self._create_embedding_summary(db_record, level_data)
                                metadata = {
                                    'float_id': str(db_record.get('float_id', 'unknown')),
                                    'lat': db_record.get('lat'),
                                    'lon': db_record.get('lon'),
                                    'time': db_record.get('time_start'),
                                    'parquet': parquet_path,
                                    'variables': list(level_data.keys()) if level_data else []
                                }
                                
                                self.emb.add_documents(
                                    [db_record['profile_id']],
                                    [metadata],
                                    [summary]
                                )
                            except Exception as e:
                                logger.warning(f"Could not create embedding: {e}")
                        
                        successful_profiles += 1
                        logger.debug(f"Successfully ingested profile {profile_idx + 1}/{total_profiles}")
                
                except Exception as e:
                    logger.warning(f"Error processing profile {profile_idx}: {e}")
                    continue
            
            return successful_profiles, total_profiles
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return 0, 0
        
        finally:
            if ds is not None:
                ds.close()

def find_all_netcdf_files(directory):
    """Find all NetCDF files with various extensions"""
    directory = Path(directory)
    extensions = ['*.nc', '*.NC', '*.netcdf', '*.NETCDF', '*.cdf', '*.CDF']
    
    files = []
    for ext in extensions:
        files.extend(directory.rglob(ext))
    
    return [str(f) for f in sorted(files)]

def run_ultra_robust_ingestion(data_dir, parquet_dir=None, max_files=None):
    """Run the ultra-robust ingestion process"""
    
    if parquet_dir is None:
        parquet_dir = PARQUET_DIR
    
    print(f"🌊 Ultra-Robust ARGO Ingestion Starting...")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Parquet directory: {parquet_dir}")
    print("=" * 60)
    
    # Initialize ingestor
    try:
        ingestor = DynamicArgoIngestor(parquet_dir)
        print("✅ Dynamic ingestor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize ingestor: {e}")
        return False
    
    # Find files
    try:
        files = find_all_netcdf_files(data_dir)
        if max_files:
            files = files[:max_files]
        print(f"📁 Found {len(files)} NetCDF files")
    except Exception as e:
        print(f"❌ Error finding files: {e}")
        return False
    
    if not files:
        print(f"❌ No NetCDF files found in {data_dir}")
        return False
    
    # Process files
    total_files = len(files)
    total_profiles_found = 0
    total_profiles_ingested = 0
    failed_files = []
    
    print(f"\n🔄 Processing {total_files} files...")
    
    for i, file_path in enumerate(tqdm(files, desc="Ingesting files")):
        try:
            successful, total = ingestor.process_file(file_path)
            total_profiles_found += total
            total_profiles_ingested += successful
            
            if total == 0:
                failed_files.append(file_path)
            
            # Progress update
            if (i + 1) % 5 == 0 or i == total_files - 1:
                logger.info(f"Progress: {i+1}/{total_files} files, "
                           f"{total_profiles_ingested} profiles ingested")
        
        except Exception as e:
            logger.error(f"Critical error with {file_path}: {e}")
            failed_files.append(file_path)
    
    # Final summary
    print(f"\n🎯 INGESTION COMPLETE!")
    print("=" * 40)
    print(f"📊 Files processed: {total_files}")
    print(f"❌ Files failed: {len(failed_files)}")
    print(f"📈 Profiles found: {total_profiles_found}")
    print(f"✅ Profiles ingested: {total_profiles_ingested}")
    
    if total_profiles_found > 0:
        success_rate = (total_profiles_ingested / total_profiles_found) * 100
        print(f"📊 Success rate: {success_rate:.1f}%")
    
    if failed_files:
        print(f"\n⚠️ Failed files ({len(failed_files)}):")
        for f in failed_files[:5]:  # Show first 5 failed files
            print(f"   {Path(f).name}")
        if len(failed_files) > 5:
            print(f"   ... and {len(failed_files) - 5} more")
    
    # Verify data was actually ingested
    try:
        with ingestor.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM floats"))
            db_count = result.scalar()
            print(f"\n📊 Database verification: {db_count} records in database")
            
            if db_count == 0:
                print("❌ WARNING: No data in database despite processing files!")
                return False
            else:
                print("✅ Database successfully populated")
    except Exception as e:
        print(f"❌ Could not verify database: {e}")
        return False
    
    return total_profiles_ingested > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-robust ARGO NetCDF ingestion")
    parser.add_argument("--dir", required=True, help="Directory containing NetCDF files")
    parser.add_argument("--parquet-dir", default=PARQUET_DIR, help="Output directory for parquet files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not Path(args.dir).exists():
        print(f"Input directory {args.dir} does not exist")
        exit(1)
    
    success = run_ultra_robust_ingestion(args.dir, args.parquet_dir, args.max_files)
    print(f"\n🏁 Ingestion {'SUCCESS' if success else 'FAILED'}")
    exit(0 if success else 1)