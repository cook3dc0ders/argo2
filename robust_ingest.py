# robust_ingest.py
"""
Enhanced ARGO NetCDF ingestion with dynamic schema detection.
Handles varying column names and structures across different ARGO files.
"""
import argparse
import os
from pathlib import Path
import json
import pandas as pd
import xarray as xr
from datetime import datetime
from sqlalchemy import create_engine, text, inspect
from tqdm import tqdm
from embeddings_utils import EmbeddingManager
from config import PARQUET_DIR, PG
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pg_engine():
    """Create PostgreSQL engine"""
    url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
    return create_engine(url, pool_pre_ping=True)

class ArgoSchemaDetector:
    """
    Dynamic schema detector for ARGO NetCDF files.
    Maps common variations to standardized column names.
    """
    
    # Standard variable mappings - handles different naming conventions
    VARIABLE_MAPPINGS = {
        # Temperature variants
        'temperature': ['TEMP', 'TEMPERATURE', 'TEMP_ADJUSTED', 'TEMP_QC', 'temp', 'temperature'],
        'salinity': ['PSAL', 'SALINITY', 'PSAL_ADJUSTED', 'PSAL_QC', 'psal', 'salinity', 'sal'],
        'pressure': ['PRES', 'PRESSURE', 'PRES_ADJUSTED', 'pres', 'pressure', 'depth', 'DEPTH'],
        'oxygen': ['DOXY', 'OXYGEN', 'DOXY_ADJUSTED', 'doxy', 'oxygen', 'O2'],
        'chlorophyll': ['CHLA', 'CHLOROPHYLL', 'CHLA_ADJUSTED', 'chla', 'chlorophyll', 'CHL'],
        'nitrate': ['NITRATE', 'NO3', 'nitrate', 'no3'],
        'ph': ['PH_IN_SITU_TOTAL', 'PH', 'ph'],
        'turbidity': ['TURBIDITY', 'turbidity', 'TURB']
    }
    
    # Coordinate mappings
    COORDINATE_MAPPINGS = {
        'latitude': ['LATITUDE', 'latitude', 'lat', 'LAT', 'LATITUDE_DEGREES'],
        'longitude': ['LONGITUDE', 'longitude', 'lon', 'LON', 'LONGITUDE_DEGREES'],
        'time': ['JULD', 'TIME', 'time', 'DATE_TIME', 'DATE', 'REFERENCE_DATE_TIME'],
        'platform': ['PLATFORM_NUMBER', 'platform_number', 'FLOAT_ID', 'float_id', 'WMO_INST_TYPE']
    }
    
    # Profile dimension mappings
    PROFILE_DIMS = ['N_PROF', 'N_PROFILES', 'profile', 'PROFILE']
    LEVEL_DIMS = ['N_LEVELS', 'N_LEVEL', 'level', 'LEVEL', 'depth', 'DEPTH']
    
    def __init__(self):
        self.detected_schemas = {}  # Cache for file schemas
    
    def detect_schema(self, ds):
        """Detect the schema of an xarray dataset"""
        schema = {
            'variables': {},
            'coordinates': {},
            'dimensions': {},
            'profile_structure': 'unknown'
        }
        
        # Detect dimensions
        dims = list(ds.dims.keys())
        schema['dimensions'] = dims
        
        # Detect profile structure
        for prof_dim in self.PROFILE_DIMS:
            if prof_dim in dims:
                schema['profile_structure'] = 'multi_profile'
                schema['profile_dim'] = prof_dim
                schema['n_profiles'] = ds.sizes[prof_dim]
                break
        else:
            schema['profile_structure'] = 'single_profile'
            schema['n_profiles'] = 1
        
        # Detect level dimension
        for level_dim in self.LEVEL_DIMS:
            if level_dim in dims:
                schema['level_dim'] = level_dim
                break
        
        # Detect variables
        for standard_name, variants in self.VARIABLE_MAPPINGS.items():
            for variant in variants:
                if variant in ds.variables:
                    schema['variables'][standard_name] = variant
                    break
        
        # Detect coordinates
        for coord_type, variants in self.COORDINATE_MAPPINGS.items():
            for variant in variants:
                if variant in ds.variables:
                    schema['coordinates'][coord_type] = variant
                    break
        
        return schema
    
    def extract_variable_safely(self, ds, var_name, profile_idx=None, schema=None):
        """Safely extract a variable, handling different data types and structures"""
        if var_name not in ds.variables:
            return None
        
        try:
            var = ds[var_name]
            
            # Handle profile indexing
            if profile_idx is not None and schema and schema['profile_structure'] == 'multi_profile':
                if schema['profile_dim'] in var.dims:
                    var = var.isel({schema['profile_dim']: profile_idx})
            
            # Extract values
            values = var.values
            
            # Handle different data types
            if hasattr(values, 'item') and values.ndim == 0:
                # Scalar value
                result = values.item()
            elif hasattr(values, 'tobytes') and values.dtype.char == 'S':
                # String/bytes
                result = values.tobytes().decode('utf-8', errors='ignore').strip('\x00 ')
            elif hasattr(values, 'astype') and values.dtype.kind in 'fc':
                # Numeric array - handle conversion properly
                try:
                    # Convert to float, handling invalid values
                    if values.ndim == 0:
                        result = float(values.item())
                    else:
                        # For arrays, convert each element safely
                        result = []
                        flat_values = values.flatten()
                        for val in flat_values:
                            try:
                                if np.isnan(val) or np.isinf(val):
                                    result.append(None)
                                else:
                                    result.append(float(val))
                            except (ValueError, TypeError):
                                result.append(None)
                except (ValueError, TypeError, OverflowError):
                    # If conversion fails, try to extract raw values
                    result = values.tolist() if hasattr(values, 'tolist') else values
            else:
                # For other data types, try direct conversion
                try:
                    if hasattr(values, 'tolist'):
                        result = values.tolist()
                    elif hasattr(values, 'item') and values.size == 1:
                        result = values.item()
                    else:
                        result = values
                except:
                    result = values
            
            # Clean string results
            if isinstance(result, str):
                result = result.replace('\x00', '').strip()
                if result == '' or result.lower() == 'nan':
                    result = None
            
            return result
            
        except Exception as e:
            logger.warning(f"Error extracting {var_name}: {e}")
            return None

def find_netcdf_files(nc_dir):
    """Find NetCDF files with multiple possible extensions"""
    nc_dir = Path(nc_dir)
    if not nc_dir.exists():
        raise FileNotFoundError(f"Directory {nc_dir} does not exist")
    
    patterns = ["**/*.nc", "**/*.NC", "**/*.netcdf", "**/*.NETCDF", "**/*.cdf", "**/*.CDF"]
    files = []
    
    for pattern in patterns:
        files.extend(list(nc_dir.glob(pattern)))
    
    return sorted([str(f) for f in files])

def parse_argo_netcdf_robust(path):
    """
    Robust parser that handles varying ARGO NetCDF schemas.
    Uses the ArgoSchemaDetector to adapt to different file structures.
    """
    detector = ArgoSchemaDetector()
    
    try:
        # Try different decode approaches for problematic files
        ds = None
        for decode_times in [False, True]:  # Try without decoding times first
            try:
                ds = xr.open_dataset(path, decode_times=decode_times, mask_and_scale=True)
                break
            except Exception as e:
                logger.warning(f"Failed to open {path} with decode_times={decode_times}: {e}")
                continue
        
        if ds is None:
            logger.error(f"Cannot open {path} with any method")
            return []
        
        # Detect schema
        schema = detector.detect_schema(ds)
        logger.info(f"Detected schema for {Path(path).name}: {schema['profile_structure']}, "
                   f"{schema['n_profiles']} profiles, variables: {list(schema['variables'].keys())}")
        
        profiles = []
        
        # Extract profiles based on detected structure
        if schema['profile_structure'] == 'multi_profile':
            for ip in range(schema['n_profiles']):
                profile = extract_profile_with_schema(ds, schema, ip, path)
                if profile:
                    profiles.append(profile)
        else:
            profile = extract_profile_with_schema(ds, schema, None, path)
            if profile:
                profiles.append(profile)
        
        return profiles
        
    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return []
    finally:
        if ds is not None:
            ds.close()

def extract_profile_with_schema(ds, schema, profile_idx, source_path):
    """Extract a single profile using the detected schema"""
    detector = ArgoSchemaDetector()
    
    try:
        # Extract coordinates using detected schema
        lat = None
        lon = None
        time_val = None
        platform_num = None
        
        # Extract coordinates
        if 'latitude' in schema['coordinates']:
            lat = detector.extract_variable_safely(
                ds, schema['coordinates']['latitude'], profile_idx, schema
            )
        
        if 'longitude' in schema['coordinates']:
            lon = detector.extract_variable_safely(
                ds, schema['coordinates']['longitude'], profile_idx, schema
            )
        
        if 'time' in schema['coordinates']:
            time_val = detector.extract_variable_safely(
                ds, schema['coordinates']['time'], profile_idx, schema
            )
        
        if 'platform' in schema['coordinates']:
            platform_num = detector.extract_variable_safely(
                ds, schema['coordinates']['platform'], profile_idx, schema
            )
        
        # Skip profiles without essential coordinates
        if lat is None or lon is None:
            logger.warning(f"Skipping profile due to missing coordinates in {source_path}")
            return None
        
        # Ensure coordinates are numeric
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            logger.warning(f"Invalid coordinate values in {source_path}: lat={lat}, lon={lon}")
            return None
        
        # Extract level data for all detected variables
        level_data = {}
        variables_present = []
        
        for standard_name, nc_var_name in schema['variables'].items():
            values = detector.extract_variable_safely(ds, nc_var_name, profile_idx, schema)
            if values is not None:
                # Ensure it's a list
                if not isinstance(values, list):
                    if np.isscalar(values):
                        values = [values]
                    else:
                        values = np.array(values).flatten().tolist()
                
                # Remove NaN values but keep the structure
                clean_values = []
                for v in values:
                    if pd.isna(v) or v is None:
                        clean_values.append(None)
                    else:
                        clean_values.append(float(v))
                
                level_data[standard_name.upper()] = clean_values
                variables_present.append(standard_name.upper())
        
        # Create profile ID
        profile_suffix = f"_profile_{profile_idx}" if profile_idx is not None else "_profile_0"
        profile_id = f"{Path(source_path).stem}{profile_suffix}"
        
        # Handle platform number
        if platform_num is None:
            platform_num = f"float_{Path(source_path).stem}"
        else:
            platform_num = str(platform_num)
        
        # Format time
        time_str = None
        if time_val is not None:
            try:
                if hasattr(time_val, 'strftime'):
                    time_str = time_val.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(time_val, str):
                    time_str = time_val
                else:
                    time_str = str(time_val)
            except:
                time_str = str(time_val)
        
        # Calculate number of valid levels
        n_levels = 0
        if level_data:
            # Use the variable with the most measurements
            n_levels = max(len(v) for v in level_data.values() if v)
        
        return {
            "float_id": platform_num,
            "profile_id": profile_id,
            "time": time_str,
            "lat": lat,
            "lon": lon,
            "n_levels": n_levels,
            "variables": variables_present,
            "level_data": level_data,
            "source_file": str(source_path),
            "detected_schema": schema  # Include schema info for debugging
        }
        
    except Exception as e:
        logger.error(f"Error extracting profile from {source_path}: {e}")
        return None

def create_flexible_database_schema():
    """
    Create a more flexible database schema that can handle varying data structures.
    """
    try:
        engine = pg_engine()
        
        # Check if table exists and needs updating
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        with engine.begin() as conn:
            if 'floats' not in existing_tables:
                # Create new flexible table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS floats (
                        id SERIAL PRIMARY KEY,
                        float_id TEXT NOT NULL,
                        profile_id TEXT UNIQUE NOT NULL,
                        source_file TEXT,
                        time_start TIMESTAMP,
                        time_end TIMESTAMP,
                        lat DOUBLE PRECISION,
                        lon DOUBLE PRECISION,
                        platform_number TEXT,
                        n_levels INTEGER DEFAULT 0,
                        variables JSONB,
                        parquet_path TEXT,
                        schema_info JSONB,  -- Store detected schema
                        raw_metadata JSONB,  -- Store all raw metadata
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    );
                """))
                
                # Create indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_float_geo 
                    ON floats USING GIST (ST_SetSRID(ST_Point(lon, lat), 4326))
                    WHERE lat IS NOT NULL AND lon IS NOT NULL;
                    
                    CREATE INDEX IF NOT EXISTS idx_float_id ON floats(float_id);
                    CREATE INDEX IF NOT EXISTS idx_profile_id ON floats(profile_id);
                    CREATE INDEX IF NOT EXISTS idx_time_start ON floats(time_start);
                    CREATE INDEX IF NOT EXISTS idx_variables ON floats USING GIN(variables);
                    CREATE INDEX IF NOT EXISTS idx_source_file ON floats(source_file);
                    CREATE INDEX IF NOT EXISTS idx_schema_info ON floats USING GIN(schema_info);
                """))
                
                logger.info("Created flexible database schema")
            else:
                # Add new columns if they don't exist
                try:
                    conn.execute(text("ALTER TABLE floats ADD COLUMN IF NOT EXISTS schema_info JSONB"))
                    conn.execute(text("ALTER TABLE floats ADD COLUMN IF NOT EXISTS raw_metadata JSONB"))
                    conn.execute(text("ALTER TABLE floats ADD COLUMN IF NOT EXISTS source_file TEXT"))
                    logger.info("Updated existing table with new columns")
                except Exception as e:
                    logger.warning(f"Could not add new columns: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up database schema: {e}")
        return False

def write_parquet_flexible(profile, parquet_root):
    """
    Write parquet file with flexible column handling.
    Creates standardized column names regardless of source schema.
    """
    try:
        # Clean profile ID for file naming
        pid = profile["profile_id"].replace("/", "_").replace(":", "_").replace("\\", "_")
        
        # Clean float ID for directory naming
        float_id_clean = str(profile["float_id"]).replace("/", "_").replace(":", "_").replace("\\", "_")
        float_id_clean = "".join(c for c in float_id_clean if c.isalnum() or c in "_-")
        
        # Create directory structure
        out_dir = Path(parquet_root) / float_id_clean
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame with standardized columns
        df_data = {}
        
        # Extract level data with standardized naming
        level_data = profile.get("level_data", {})
        
        for standard_name, values in level_data.items():
            if values and len(values) > 0:
                # Pad shorter arrays with NaN to match longest array
                df_data[standard_name] = values
        
        if not df_data:
            logger.warning(f"No valid level data for profile {profile['profile_id']}")
            return None
        
        # Create DataFrame and handle unequal lengths
        max_length = max(len(v) for v in df_data.values())
        
        for col, values in df_data.items():
            if len(values) < max_length:
                # Pad with NaN
                values.extend([np.nan] * (max_length - len(values)))
                df_data[col] = values
        
        df = pd.DataFrame(df_data)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        if df.empty:
            logger.warning(f"No valid data after cleaning for profile {profile['profile_id']}")
            return None
        
        # Save parquet file
        parquet_path = out_dir / f"{pid}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        logger.debug(f"Saved parquet: {parquet_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
        return str(parquet_path)
        
    except Exception as e:
        logger.error(f"Error writing parquet for profile {profile.get('profile_id', 'unknown')}: {e}")
        return None

def robust_ingest_file(file_path, parquet_dir, engine, emb):
    """
    Robustly ingest a single NetCDF file with error handling.
    """
    try:
        profiles = parse_argo_netcdf_robust(file_path)
        
        if not profiles:
            logger.warning(f"No profiles extracted from {file_path}")
            return 0, 0
        
        successful = 0
        total = len(profiles)
        
        for profile in profiles:
            try:
                # Validate profile has required data
                if not profile.get("level_data"):
                    logger.warning(f"Skipping profile {profile['profile_id']} - no level data")
                    continue
                
                # Write parquet
                parquet_path = write_parquet_flexible(profile, parquet_dir)
                if not parquet_path:
                    logger.warning(f"Failed to write parquet for profile {profile['profile_id']}")
                    continue
                
                # Prepare database record with proper type handling
                db_record = {
                    "float_id": str(profile.get("float_id", "unknown"))[:50],  # Limit length
                    "profile_id": str(profile["profile_id"])[:100],  # Limit length
                    "source_file": str(profile["source_file"])[:255],  # Limit length
                    "time_start": profile.get("time"),
                    "time_end": profile.get("time"),
                    "lat": float(profile["lat"]) if profile.get("lat") is not None else None,
                    "lon": float(profile["lon"]) if profile.get("lon") is not None else None,
                    "platform_number": str(profile.get("float_id", "unknown"))[:50],
                    "n_levels": int(profile["n_levels"]) if profile.get("n_levels") else 0,
                    "variables": json.dumps(profile["variables"]) if profile.get("variables") else "[]",
                    "parquet_path": str(parquet_path),
                    "schema_info": json.dumps(profile.get("detected_schema", {})),
                    "raw_metadata": json.dumps({
                        k: v for k, v in profile.items() 
                        if k not in ["level_data", "detected_schema"] and v is not None
                    })
                }
                
                # Clean up any potential problematic values
                for key, value in db_record.items():
                    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                        db_record[key] = None
                
                # Insert into database with proper error handling
                try:
                    with engine.begin() as conn:
                        sql = text("""
                            INSERT INTO floats (
                                float_id, profile_id, source_file, time_start, time_end, 
                                lat, lon, platform_number, n_levels, variables, 
                                parquet_path, schema_info, raw_metadata
                            ) VALUES (
                                :float_id, :profile_id, :source_file, :time_start, :time_end,
                                :lat, :lon, :platform_number, :n_levels, :variables::jsonb,
                                :parquet_path, :schema_info::jsonb, :raw_metadata::jsonb
                            )
                            ON CONFLICT (profile_id) DO UPDATE SET
                                parquet_path = EXCLUDED.parquet_path,
                                schema_info = EXCLUDED.schema_info,
                                raw_metadata = EXCLUDED.raw_metadata,
                                updated_at = NOW()
                        """)
                        
                        conn.execute(sql, db_record)
                        logger.debug(f"Database record inserted for {profile['profile_id']}")
                        
                except Exception as db_e:
                    logger.error(f"Database insertion failed for {profile['profile_id']}: {db_e}")
                    logger.debug(f"Failed DB record: {db_record}")
                    continue
                
                # Create embedding
                try:
                    summary = create_profile_summary(profile)
                    metadata = {
                        "float_id": str(profile["float_id"]),
                        "lat": float(profile["lat"]),
                        "lon": float(profile["lon"]),
                        "time": profile.get("time"),
                        "parquet": parquet_path,
                        "variables": profile["variables"]
                    }
                    
                    emb.add_documents([profile["profile_id"]], [metadata], [summary])
                    logger.debug(f"Embedding created for {profile['profile_id']}")
                    
                except Exception as emb_e:
                    logger.error(f"Embedding creation failed for {profile['profile_id']}: {emb_e}")
                    # Don't continue here - we already saved to DB and parquet
                
                successful += 1
                logger.info(f"Successfully ingested profile {profile['profile_id']} ({successful}/{total})")
                
            except Exception as e:
                logger.error(f"Error ingesting profile {profile.get('profile_id', 'unknown')}: {e}")
                continue
        
        return successful, total
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return 0, 0
    
    
def create_profile_summary(profile):
    """Create a natural language summary of the profile for embeddings"""
    try:
        summary_parts = [
            f"Float {profile['float_id']} profile {profile['profile_id']}:",
            f"Located at {profile['lat']:.2f}°N, {profile['lon']:.2f}°E"
        ]
        
        if profile.get('time'):
            summary_parts.append(f"measured on {profile['time']}")
        
        if profile['n_levels'] > 0:
            summary_parts.append(f"with {profile['n_levels']} depth levels")
        
        if profile['variables']:
            summary_parts.append(f"measuring {', '.join(profile['variables'])}")
        
        return " ".join(summary_parts) + "."
        
    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        return f"Profile {profile.get('profile_id', 'unknown')}"

def ingest_directory_robust(nc_dir, parquet_dir, max_files=None):
    """
    Robust ingestion function that handles varying schemas across files.
    """
    print(f"Starting robust ingestion from {nc_dir} to {parquet_dir}")
    
    # Initialize components
    try:
        engine = pg_engine()
        print("Connected to PostgreSQL")
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        return False
    
    # Set up flexible database schema
    if not create_flexible_database_schema():
        print("Failed to set up database schema")
        return False
    
    try:
        emb = EmbeddingManager()
        print("Initialized embedding manager")
    except Exception as e:
        print(f"Failed to initialize embeddings: {e}")
        return False
    
    # Find files
    try:
        files = find_netcdf_files(nc_dir)
        if max_files:
            files = files[:max_files]
        print(f"Found {len(files)} NetCDF files")
    except Exception as e:
        print(f"Error finding files: {e}")
        return False
    
    if len(files) == 0:
        print(f"No NetCDF files found in {nc_dir}")
        return False
    
    # Ensure parquet directory exists
    Path(parquet_dir).mkdir(parents=True, exist_ok=True)
    
    # Process files with progress tracking
    total_files = len(files)
    total_profiles_processed = 0
    total_profiles_successful = 0
    failed_files = []
    
    for i, file_path in enumerate(tqdm(files, desc="Processing files")):
        try:
            successful, total = robust_ingest_file(file_path, parquet_dir, engine, emb)
            total_profiles_processed += total
            total_profiles_successful += successful
            
            if total == 0:
                failed_files.append(file_path)
            
            # Progress update every 10 files
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{total_files} files, "
                           f"{total_profiles_successful}/{total_profiles_processed} profiles successful")
                
        except Exception as e:
            logger.error(f"Critical error processing {file_path}: {e}")
            failed_files.append(file_path)
            continue
    
    # Final summary
    print(f"\nIngestion complete!")
    print(f"   Files processed: {total_files}")
    print(f"   Files failed: {len(failed_files)}")
    print(f"   Total profiles found: {total_profiles_processed}")
    print(f"   Successfully ingested: {total_profiles_successful}")
    print(f"   Success rate: {total_profiles_successful/total_profiles_processed*100:.1f}%" if total_profiles_processed > 0 else "   Success rate: 0%")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files[:10]:  # Show first 10 failed files
            print(f"   {f}")
        if len(failed_files) > 10:
            print(f"   ... and {len(failed_files) - 10} more")
    
    return total_profiles_successful > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust ARGO NetCDF ingestion with schema detection")
    parser.add_argument("--dir", required=True, help="Directory containing NetCDF files")
    parser.add_argument("--parquet-dir", default=PARQUET_DIR, help="Output directory for parquet files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not Path(args.dir).exists():
        print(f"Input directory {args.dir} does not exist")
        exit(1)
    
    success = ingest_directory_robust(args.dir, args.parquet_dir, args.max_files)
    exit(0 if success else 1)