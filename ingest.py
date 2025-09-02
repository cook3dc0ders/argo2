# ingest.py
"""
Ingest ARGO NetCDF files (or a directory) and:
 - extract per-profile metadata,
 - write Parquet chunks,
 - insert metadata rows into Postgres,
 - add embeddings (metadata+short summary) into Chroma.

Usage:
 python ingest.py --dir /path/to/nc_files --parquet-dir ./parquet_store
"""
import argparse
import os
from pathlib import Path
import json
import pandas as pd
import xarray as xr
from datetime import datetime
from sqlalchemy import create_engine, text
from tqdm import tqdm
from embeddings_utils import EmbeddingManager
from config import PARQUET_DIR, PG
import numpy as np

def pg_engine():
    """Create PostgreSQL engine"""
    url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
    return create_engine(url, pool_pre_ping=True)

def find_netcdf_files(nc_dir):
    """Find NetCDF files with multiple possible extensions and case variations"""
    nc_dir = Path(nc_dir)
    if not nc_dir.exists():
        raise FileNotFoundError(f"Directory {nc_dir} does not exist")
    
    patterns = ["**/*.nc", "**/*.NC", "**/*.netcdf", "**/*.NETCDF"]
    files = []
    
    for pattern in patterns:
        files.extend(list(nc_dir.glob(pattern)))
    
    return sorted([str(f) for f in files])

def parse_argo_netcdf(path):
    """
    Parse a single ARGO NetCDF file and return a list of profile dicts.
    This function handles classic ARGO profile NetCDF structures.
    """
    try:
        ds = xr.open_dataset(path, decode_times=True, mask_and_scale=True)
    except Exception as e:
        print(f"Failed to open {path}: {e}")
        raise
    
    profiles = []
    
    try:
        # Check for multi-profile dimension - use .sizes instead of .dims for newer xarray
        nprof = ds.sizes.get("N_PROF") or ds.sizes.get("N_PROFILES") or 1
        
        if nprof > 1 and "N_PROF" in ds.sizes:
            # Multi-profile file
            for ip in range(nprof):
                profile = extract_single_profile(ds, ip, path)
                if profile:
                    profiles.append(profile)
        else:
            # Single profile file
            profile = extract_single_profile(ds, 0, path, single_profile=True)
            if profile:
                profiles.append(profile)
                
    except Exception as e:
        print(f"Error processing profiles in {path}: {e}")
    finally:
        ds.close()
    
    return profiles

def extract_single_profile(ds, profile_idx, source_path, single_profile=False):
    """Extract a single profile from the dataset"""
    try:
        # Extract coordinates and time
        if single_profile:
            lat = float(ds["LATITUDE"].values) if "LATITUDE" in ds.variables else None
            lon = float(ds["LONGITUDE"].values) if "LONGITUDE" in ds.variables else None
            juld = ds["JULD"].values if "JULD" in ds.variables else None
        else:
            lat = float(ds["LATITUDE"][profile_idx].values) if "LATITUDE" in ds.variables else None
            lon = float(ds["LONGITUDE"][profile_idx].values) if "LONGITUDE" in ds.variables else None
            juld = ds["JULD"][profile_idx].values if "JULD" in ds.variables else None
        
        # Skip if missing essential coordinates
        if lat is None or lon is None:
            return None
        
        # Extract available variables
        possible_vars = ["TEMP", "PSAL", "DOXY", "CHLA", "PRES"]
        variables = [v for v in possible_vars if v in ds.variables]
        
        if not variables:
            print(f"No recognized variables in profile {profile_idx} of {source_path}")
            return None
        
        # Extract level data
        level_vars = {}
        for var in variables:
            try:
                if single_profile:
                    arr = ds[var].values
                else:
                    arr = ds[var].isel(N_PROF=profile_idx).values
                
                # Convert to list, handling NaN values
                if hasattr(arr, 'tolist'):
                    level_vars[var] = arr.tolist()
                else:
                    level_vars[var] = [float(arr)] if np.isscalar(arr) else []
            except Exception as e:
                print(f"Error extracting {var}: {e}")
                level_vars[var] = []
        
        # Create profile ID
        profile_id = f"{Path(source_path).stem}::profile_{profile_idx}"
        
        # Get platform number - FIX: properly extract string value
        try:
            if single_profile:
                if "PLATFORM_NUMBER" in ds.variables:
                    platform_raw = ds["PLATFORM_NUMBER"].values
                    # Handle different data types
                    if hasattr(platform_raw, 'item'):
                        platform_num = str(platform_raw.item())
                    elif hasattr(platform_raw, 'decode'):
                        platform_num = platform_raw.decode('utf-8').strip()
                    else:
                        platform_num = str(platform_raw).strip()
                else:
                    platform_num = Path(source_path).stem
            else:
                if "PLATFORM_NUMBER" in ds.variables:
                    platform_raw = ds["PLATFORM_NUMBER"][profile_idx].values
                    # Handle different data types
                    if hasattr(platform_raw, 'item'):
                        platform_num = str(platform_raw.item())
                    elif hasattr(platform_raw, 'decode'):
                        platform_num = platform_raw.decode('utf-8').strip()
                    else:
                        platform_num = str(platform_raw).strip()
                else:
                    platform_num = Path(source_path).stem
            
            # Clean platform number for use as directory name
            platform_num = platform_num.replace('\x00', '').strip()
            if not platform_num or platform_num == 'nan':
                platform_num = f"float_{profile_idx:04d}"
                
        except Exception as e:
            print(f"Error extracting platform number: {e}")
            platform_num = f"float_{profile_idx:04d}"
        
        # Format time
        time_str = None
        if juld is not None:
            try:
                if hasattr(juld, 'strftime'):
                    time_str = juld.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = str(np.datetime_as_string(juld, unit='s'))
            except:
                time_str = str(juld)
        
        return {
            "float_id": platform_num,
            "profile_id": profile_id,
            "time": time_str,
            "lat": lat,
            "lon": lon,
            "n_levels": len(level_vars.get("PRES", [])),
            "variables": [v for v in variables if v != "PRES"],  # Don't include PRES in variables list
            "level_data": level_vars,
            "source_file": str(source_path)
        }
        
    except Exception as e:
        print(f"Error extracting profile {profile_idx} from {source_path}: {e}")
        return None

def write_parquet(profile, parquet_root):
    """Store the per-profile level data as parquet with metadata"""
    # Use pathlib for cross-platform path handling
    pid = profile["profile_id"].replace("/", "_").replace(":", "_").replace("\\", "_")
    
    # Clean float_id for directory name
    float_id_clean = str(profile["float_id"]).replace("/", "_").replace(":", "_").replace("\\", "_")
    float_id_clean = float_id_clean.replace("<", "").replace(">", "").replace("|", "_")
    
    # Create directory structure
    out_dir = Path(parquet_root) / float_id_clean
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame from level data
    df = pd.DataFrame(profile["level_data"])
    
    # Remove rows where all values are NaN
    df = df.dropna(how='all')
    
    if df.empty:
        print(f"Warning: No valid data for profile {profile['profile_id']}")
        return None
    
    # Save parquet file
    parquet_path = out_dir / f"{pid}.parquet"
    df.to_parquet(parquet_path, index=False)
    
    # Return string path for database storage
    return str(parquet_path)

def ingest_directory(nc_dir, parquet_dir):
    """Main ingestion function"""
    print(f"Starting ingestion from {nc_dir} to {parquet_dir}")
    
    # Initialize components
    try:
        engine = pg_engine()
        print("Connected to PostgreSQL")
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        return
    
    try:
        emb = EmbeddingManager()
        print("Initialized embedding manager")
    except Exception as e:
        print(f"Failed to initialize embeddings: {e}")
        return
    
    # Find files
    try:
        files = find_netcdf_files(nc_dir)
        print(f"Found {len(files)} NetCDF files")
    except Exception as e:
        print(f"Error finding files: {e}")
        return
    
    if len(files) == 0:
        print(f"No NetCDF files found in {nc_dir}")
        return
    
    # Ensure parquet directory exists
    Path(parquet_dir).mkdir(parents=True, exist_ok=True)
    
    # Process files
    total_profiles = 0
    successful_profiles = 0
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            profiles = parse_argo_netcdf(file_path)
            
            for profile in profiles:
                total_profiles += 1
                try:
                    # Write parquet file
                    parquet_path = write_parquet(profile, parquet_dir)
                    if parquet_path is None:
                        continue
                    
                    # Insert metadata into PostgreSQL - FIX: Use consistent parameter style
                    with engine.begin() as conn:
                        sql = text("""
                        INSERT INTO floats (float_id, profile_id, time_start, time_end, lat, lon, platform_number, n_levels, variables, parquet_path)
                        VALUES (:float_id, :profile_id, :time_start, :time_end, :lat, :lon, :platform_number, :n_levels, :variables::jsonb, :parquet_path)
                        ON CONFLICT (profile_id) DO UPDATE SET
                            parquet_path = EXCLUDED.parquet_path,
                            created_at = NOW()
                        """)
                        
                        conn.execute(sql, {
                            "float_id": profile["float_id"],
                            "profile_id": profile["profile_id"],
                            "time_start": profile["time"],
                            "time_end": profile["time"],
                            "lat": profile["lat"],
                            "lon": profile["lon"],
                            "platform_number": profile["float_id"],
                            "n_levels": profile["n_levels"],
                            "variables": json.dumps(profile["variables"]),
                            "parquet_path": parquet_path
                        })
                    
                    # Create embedding
                    summary = (f"Float {profile['float_id']} profile {profile['profile_id']}: "
                             f"{profile['n_levels']} levels at ({profile['lat']:.2f},{profile['lon']:.2f}) "
                             f"time {profile['time']}. Variables: {','.join(profile['variables'])}")
                    
                    metadata = {
                        "float_id": profile["float_id"], 
                        "lat": profile["lat"], 
                        "lon": profile["lon"], 
                        "time": profile["time"], 
                        "parquet": parquet_path
                    }
                    
                    emb.add_documents([profile["profile_id"]], [metadata], [summary])
                    successful_profiles += 1
                    
                except Exception as e:
                    print(f"Error ingesting profile {profile.get('profile_id', 'unknown')}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    print(f"\nIngestion complete!")
    print(f"   Total profiles processed: {total_profiles}")
    print(f"   Successfully ingested: {successful_profiles}")
    print(f"   Failed: {total_profiles - successful_profiles}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest ARGO NetCDF files into RAG system")
    parser.add_argument("--dir", required=True, help="Directory containing NetCDF files")
    parser.add_argument("--parquet-dir", default=PARQUET_DIR, help="Output directory for parquet files")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.dir).exists():
        print(f"Input directory {args.dir} does not exist")
        exit(1)
    
    ingest_directory(args.dir, args.parquet_dir)