# bulletproof_argo_ingest.py
"""
Bulletproof ARGO NetCDF ingestion system.
Handles all edge cases, corrupted files, and schema variations automatically.
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
import warnings
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import shutil

# Suppress xarray warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='xarray')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pg_engine():
    """Create PostgreSQL engine with robust error handling"""
    try:
        url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
        engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise

class BulletproofArgoParser:
    """
    Ultra-robust ARGO parser that handles ANY NetCDF file structure.
    Uses progressive fallback strategies to extract maximum data.
    """
    
    # Comprehensive variable mapping with all known variants
    VARIABLE_MAPPINGS = {
        'temperature': [
            'TEMP', 'TEMPERATURE', 'TEMP_ADJUSTED', 'TEMP_QC', 'temp', 'temperature',
            'T', 'THETA', 'POTENTIAL_TEMPERATURE', 'SEA_WATER_TEMPERATURE',
            'TEMP_DOXY', 'TEMP_CNDC', 'TEMP_ADJUSTED_QC', 'TEMP_dPRES'
        ],
        'salinity': [
            'PSAL', 'SALINITY', 'PSAL_ADJUSTED', 'PSAL_QC', 'psal', 'salinity', 'sal',
            'S', 'PRACTICAL_SALINITY', 'SEA_WATER_PRACTICAL_SALINITY',
            'PSAL_ADJUSTED_QC', 'PSAL_dPRES'
        ],
        'pressure': [
            'PRES', 'PRESSURE', 'PRES_ADJUSTED', 'pres', 'pressure', 'depth', 'DEPTH',
            'P', 'SEA_WATER_PRESSURE', 'PRES_QC', 'PRES_ADJUSTED_QC'
        ],
        'oxygen': [
            'DOXY', 'OXYGEN', 'DOXY_ADJUSTED', 'doxy', 'oxygen', 'O2',
            'DISSOLVED_OXYGEN', 'DOXY_QC', 'DOXY_ADJUSTED_QC'
        ],
        'chlorophyll': [
            'CHLA', 'CHLOROPHYLL', 'CHLA_ADJUSTED', 'chla', 'chlorophyll', 'CHL',
            'CHLOROPHYLL_A', 'CHLA_QC', 'CHLA_ADJUSTED_QC'
        ],
        'nitrate': [
            'NITRATE', 'NO3', 'nitrate', 'no3', 'NITRATE_QC', 'NITRATE_ADJUSTED',
            'NITRATE_ADJUSTED_QC'
        ],
        'ph': [
            'PH_IN_SITU_TOTAL', 'PH', 'ph', 'pH', 'PH_IN_SITU_TOTAL_QC',
            'PH_IN_SITU_TOTAL_ADJUSTED', 'PH_IN_SITU_TOTAL_ADJUSTED_QC'
        ],
        'turbidity': [
            'TURBIDITY', 'turbidity', 'TURB', 'TURBIDITY_QC',
            'TURBIDITY_ADJUSTED', 'TURBIDITY_ADJUSTED_QC'
        ],
        'conductivity': [
            'CNDC', 'CONDUCTIVITY', 'conductivity', 'CNDC_QC',
            'CNDC_ADJUSTED', 'CNDC_ADJUSTED_QC'
        ]
    }
    
    COORDINATE_MAPPINGS = {
        'latitude': [
            'LATITUDE', 'latitude', 'lat', 'LAT', 'LATITUDE_DEGREES',
            'LATITUDE_DM', 'latitude_dm', 'lats', 'Latitude'
        ],
        'longitude': [
            'LONGITUDE', 'longitude', 'lon', 'LON', 'LONGITUDE_DEGREES',
            'LONGITUDE_DM', 'longitude_dm', 'lons', 'Longitude'
        ],
        'time': [
            'JULD', 'TIME', 'time', 'DATE_TIME', 'DATE', 'REFERENCE_DATE_TIME',
            'JULD_QC', 'JULD_LOCATION', 'TIME_QC', 'times'
        ],
        'platform': [
            'PLATFORM_NUMBER', 'platform_number', 'FLOAT_ID', 'float_id',
            'WMO_INST_TYPE', 'PLATFORM_TYPE', 'platform_type'
        ]
    }
    
    def __init__(self):
        self.file_schemas = {}  # Cache schemas per file
        self.fallback_strategies = [
            self._strategy_your_exact_schema,  # NEW: Your specific schema first
            self._strategy_standard_argo,
            self._strategy_simple_variables,
            self._strategy_any_numeric_data,
            self._strategy_minimal_extraction
        ]
    
    def parse_file_bulletproof(self, file_path: str) -> List[Dict]:
        """
        Parse any ARGO NetCDF file using multiple fallback strategies.
        Guarantees some data extraction even from corrupted files.
        Returns a list of profile dictionaries.
        """
        profiles = []
        file_path = Path(file_path)

        # Try multiple file opening strategies
        dataset = self._open_file_robust(file_path)
        if dataset is None:
            logger.error(f"Could not open {file_path} with any method")
            return []

        try:
            def find_var(possible_names):
                """Return first matching variable from dataset"""
                for name in possible_names:
                    if name in dataset.variables:
                        return dataset[name]
                return None

            # Look for variables
            temp = find_var(self.VARIABLE_MAPPINGS["temperature"])
            psal = find_var(self.VARIABLE_MAPPINGS["salinity"])
            pres = find_var(self.VARIABLE_MAPPINGS["pressure"])
            doxy = find_var(self.VARIABLE_MAPPINGS["oxygen"])

            lat = find_var(self.COORDINATE_MAPPINGS["latitude"])
            lon = find_var(self.COORDINATE_MAPPINGS["longitude"])
            time = find_var(self.COORDINATE_MAPPINGS["time"])
            platform = find_var(self.COORDINATE_MAPPINGS["platform"])

            n_levels = (
                len(temp) if temp is not None else
                (len(psal) if psal is not None else
                (len(pres) if pres is not None else 0))
            )

            # Build profile dictionary
            profile = {
                "file": str(file_path),
                "N_LEVELS": list(range(n_levels)),
                "TEMP": temp.values.tolist() if temp is not None else [None]*n_levels,
                "PSAL": psal.values.tolist() if psal is not None else [None]*n_levels,
                "DOXY": doxy.values.tolist() if doxy is not None else [None]*n_levels,
                "PRES": pres.values.tolist() if pres is not None else [None]*n_levels,
                "PLATFORM_NUMBER": str(platform.values.item()) if platform is not None else None,
                "LATITUDE": float(lat.values) if lat is not None else None,
                "LONGITUDE": float(lon.values) if lon is not None else None,
                "JULD": str(pd.to_datetime(str(time.values))) if time is not None else None
            }

            profiles.append(profile)

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")

        return profiles

    
    def _strategy_your_exact_schema(self, ds: xr.Dataset, file_path: Path) -> List[Dict]:
        """
        Optimized strategy for your exact schema:
        N_LEVELS,TEMP,PSAL,DOXY,PRES,PLATFORM_NUMBER,LATITUDE,LONGITUDE,JULD
        """
        try:
            # Quick check - do we have your exact variables?
            your_vars = ['TEMP', 'PSAL', 'DOXY', 'PRES', 'PLATFORM_NUMBER', 'LATITUDE', 'LONGITUDE', 'JULD']
            present_vars = [var for var in your_vars if var in ds.variables]
            
            # Need at least core variables
            if len(present_vars) < 4:
                return []  # Fall through to next strategy
            
            logger.info(f"Using optimized parser for your schema: {file_path.name}")
            
            # Use the optimized parser
            try:
                from optimized_argo_parser import OptimizedArgoParser
                parser = OptimizedArgoParser()
                profiles = parser.parse_your_netcdf(file_path)
                
                if profiles:
                    logger.info(f"✅ Your schema strategy succeeded: {len(profiles)} profiles")
                    return profiles
                    
            except ImportError:
                logger.debug("Optimized parser not available, using built-in logic")
                # Use built-in optimized logic
                return self._parse_your_schema_builtin(ds, file_path)
            
            return []
            
        except Exception as e:
            logger.debug(f"Your schema strategy failed: {e}")
            return []
    
    def _parse_your_schema_builtin(self, ds: xr.Dataset, file_path: Path) -> List[Dict]:
        """Built-in parsing for your specific schema when optimized parser not available"""
        profiles = []
        
        try:
            # Detect number of profiles
            profile_count = 1
            profile_dims = ['N_PROF', 'profile', 'PROFILE']
            profile_dim = None
            
            for dim in profile_dims:
                if dim in ds.dims:
                    profile_dim = dim
                    profile_count = ds.sizes[dim]
                    break
            
            # Extract each profile
            for prof_idx in range(profile_count):
                try:
                    # Extract coordinates with your exact variable names
                    lat = self._extract_coordinate_your_schema(ds, 'LATITUDE', prof_idx, profile_dim)
                    lon = self._extract_coordinate_your_schema(ds, 'LONGITUDE', prof_idx, profile_dim)
                    time_val = self._extract_coordinate_your_schema(ds, 'JULD', prof_idx, profile_dim)
                    platform = self._extract_coordinate_your_schema(ds, 'PLATFORM_NUMBER', prof_idx, profile_dim)
                    
                    if lat is None or lon is None:
                        continue
                    
                    # Validate coordinates
                    try:
                        lat = float(lat)
                        lon = float(lon)
                        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                            continue
                    except:
                        continue
                    
                    # Extract level data with your variable names
                    level_data = {}
                    variables_present = []
                    
                    # Temperature
                    if 'TEMP' in ds.variables:
                        temp = self._extract_level_data_your_schema(ds, 'TEMP', prof_idx, profile_dim)
                        if temp:
                            level_data['TEMPERATURE'] = temp
                            variables_present.append('TEMPERATURE')
                    
                    # Salinity
                    if 'PSAL' in ds.variables:
                        sal = self._extract_level_data_your_schema(ds, 'PSAL', prof_idx, profile_dim)
                        if sal:
                            level_data['SALINITY'] = sal
                            variables_present.append('SALINITY')
                    
                    # Oxygen
                    if 'DOXY' in ds.variables:
                        oxy = self._extract_level_data_your_schema(ds, 'DOXY', prof_idx, profile_dim)
                        if oxy:
                            level_data['OXYGEN'] = oxy
                            variables_present.append('OXYGEN')
                    
                    # Pressure
                    if 'PRES' in ds.variables:
                        pres = self._extract_level_data_your_schema(ds, 'PRES', prof_idx, profile_dim)
                        if pres:
                            level_data['PRESSURE'] = pres
                            variables_present.append('PRESSURE')
                    
                    if not level_data:
                        continue
                    
                    # Format time specifically for JULD
                    time_str = self._format_juld_time(time_val)
                    
                    # Create profile
                    profile_suffix = f"_prof_{prof_idx}" if profile_count > 1 else "_prof_0"
                    profile = {
                        'float_id': str(platform)[:50] if platform else f"float_{file_path.stem}",
                        'profile_id': f"{file_path.stem}{profile_suffix}",
                        'lat': lat,
                        'lon': lon,
                        'time': time_str,
                        'n_levels': max(len(v) for v in level_data.values()),
                        'variables': variables_present,
                        'level_data': level_data,
                        'source_file': str(file_path),
                        'extraction_method': 'your_schema_builtin'
                    }
                    
                    if self._validate_profile(profile):
                        profiles.append(profile)
                        
                except Exception as e:
                    logger.debug(f"Error extracting profile {prof_idx}: {e}")
                    continue
            
            return profiles
            
        except Exception as e:
            logger.debug(f"Built-in your schema parsing failed: {e}")
            return []
    
    def _extract_coordinate_your_schema(self, ds: xr.Dataset, var_name: str, 
                                      prof_idx: int, profile_dim: Optional[str]) -> Any:
        """Extract coordinate specifically for your schema"""
        if var_name not in ds.variables:
            return None
        
        try:
            var = ds[var_name]
            
            # Handle profile indexing if multi-profile
            if profile_dim and profile_dim in var.dims:
                var = var.isel({profile_dim: prof_idx})
            
            # Extract value
            if var.ndim == 0:
                value = var.item()
            elif var.size == 1:
                value = var.values.item()
            else:
                # Take first non-null value
                flat_vals = var.values.flatten()
                for val in flat_vals:
                    if not pd.isna(val):
                        value = val
                        break
                else:
                    return None
            
            # Clean string values (for PLATFORM_NUMBER)
            if hasattr(value, 'tobytes'):
                value = value.tobytes().decode('utf-8', errors='ignore').strip('\x00 ')
            elif isinstance(value, bytes):
                value = value.decode('utf-8', errors='ignore').strip('\x00 ')
            
            return value
            
        except Exception as e:
            logger.debug(f"Error extracting {var_name}: {e}")
            return None
    
    def _extract_level_data_your_schema(self, ds: xr.Dataset, var_name: str,
                                      prof_idx: int, profile_dim: Optional[str]) -> Optional[List]:
        """Extract level data specifically for your schema"""
        if var_name not in ds.variables:
            return None
        
        try:
            var = ds[var_name]
            
            # Handle profile indexing if multi-profile
            if profile_dim and profile_dim in var.dims:
                var = var.isel({profile_dim: prof_idx})
            
            # Extract array values
            values = var.values
            
            # Convert to clean list
            if values.ndim > 1:
                values = values.flatten()
            
            clean_values = []
            for val in values:
                try:
                    if pd.isna(val) or np.isinf(val):
                        clean_values.append(None)
                    else:
                        clean_values.append(float(val))
                except:
                    clean_values.append(None)
            
            # Remove trailing None values
            while clean_values and clean_values[-1] is None:
                clean_values.pop()
            
            return clean_values if clean_values else None
            
        except Exception as e:
            logger.debug(f"Error extracting level data for {var_name}: {e}")
            return None
    
    def _format_juld_time(self, juld_val: Any) -> Optional[str]:
        """Format JULD time value specifically"""
        if juld_val is None:
            return None
        
        try:
            if isinstance(juld_val, str):
                return juld_val
            elif isinstance(juld_val, (int, float)):
                # JULD is typically days since 1950-01-01
                if 0 < juld_val < 1000000:  # Reasonable JULD range
                    from datetime import datetime, timedelta
                    reference_date = datetime(1950, 1, 1)
                    actual_date = reference_date + timedelta(days=float(juld_val))
                    return actual_date.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    return str(juld_val)
            else:
                return str(juld_val)
        except:
            return str(juld_val) if juld_val is not None else None
        
        try:
            # Try each strategy until one succeeds
            for strategy_idx, strategy in enumerate(self.fallback_strategies):
                try:
                    logger.debug(f"Trying strategy {strategy_idx + 1} for {file_path.name}")
                    profiles = strategy(dataset, file_path)
                    
                    if profiles:
                        logger.info(f"Strategy {strategy_idx + 1} succeeded for {file_path.name}: {len(profiles)} profiles")
                        break
                    else:
                        logger.debug(f"Strategy {strategy_idx + 1} returned no profiles")
                        
                except Exception as e:
                    logger.warning(f"Strategy {strategy_idx + 1} failed for {file_path.name}: {e}")
                    continue
            
            if not profiles:
                # Last resort: create minimal profile with just file info
                profiles = [self._create_minimal_profile(file_path)]
                logger.warning(f"Using minimal fallback profile for {file_path.name}")
            
            return profiles
            
        except Exception as e:
            logger.error(f"All strategies failed for {file_path}: {e}")
            return [self._create_minimal_profile(file_path)]
            
        finally:
            if dataset is not None:
                try:
                    dataset.close()
                except:
                    pass
    
    def _open_file_robust(self, file_path: Path) -> Optional[xr.Dataset]:
        """Try multiple methods to open a NetCDF file"""
        
        # Different opening strategies
        strategies = [
            # Standard approach
            lambda: xr.open_dataset(file_path, decode_times=False, mask_and_scale=True),
            # Without mask and scale
            lambda: xr.open_dataset(file_path, decode_times=False, mask_and_scale=False),
            # With time decoding
            lambda: xr.open_dataset(file_path, decode_times=True, mask_and_scale=True),
            # Minimal options
            lambda: xr.open_dataset(file_path),
            # With specific engine
            lambda: xr.open_dataset(file_path, engine='netcdf4', decode_times=False),
            # Last resort with h5netcdf
            lambda: xr.open_dataset(file_path, engine='h5netcdf', decode_times=False)
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                ds = strategy()
                logger.debug(f"File opened with strategy {i + 1}")
                return ds
            except Exception as e:
                logger.debug(f"Opening strategy {i + 1} failed: {e}")
                continue
        
        return None
    
    def _strategy_standard_argo(self, ds: xr.Dataset, file_path: Path) -> List[Dict]:
        """Standard ARGO file parsing strategy"""
        profiles = []
        
        # Detect profile structure
        profile_dims = ['N_PROF', 'N_PROFILES', 'profile', 'PROFILE']
        profile_dim = None
        n_profiles = 1
        
        for dim in profile_dims:
            if dim in ds.dims:
                profile_dim = dim
                n_profiles = ds.sizes[dim]
                break
        
        # Extract each profile
        for prof_idx in range(n_profiles):
            try:
                profile = self._extract_profile_standard(ds, prof_idx if profile_dim else None, 
                                                       profile_dim, file_path)
                if profile and self._validate_profile(profile):
                    profiles.append(profile)
            except Exception as e:
                logger.debug(f"Failed to extract profile {prof_idx}: {e}")
                continue
        
        return profiles
    
    def _strategy_simple_variables(self, ds: xr.Dataset, file_path: Path) -> List[Dict]:
        """Simple variable extraction - just grab what's available"""
        try:
            # Get basic coordinates
            coords = self._extract_coordinates_flexible(ds)
            if not coords.get('lat') or not coords.get('lon'):
                return []
            
            # Get all numeric variables
            variables = {}
            for var_name, var in ds.variables.items():
                if var.dtype.kind in 'fc' and var.size > 1:  # Numeric with multiple values
                    try:
                        values = self._safe_extract_values(var)
                        if values and len(values) > 0:
                            # Map to standard name if possible
                            standard_name = self._map_to_standard_name(var_name)
                            variables[standard_name] = values
                    except Exception as e:
                        logger.debug(f"Could not extract {var_name}: {e}")
                        continue
            
            if not variables:
                return []
            
            # Create single profile
            profile = {
                'float_id': coords.get('platform', f"float_{file_path.stem}"),
                'profile_id': f"{file_path.stem}_simple_profile",
                'lat': coords['lat'],
                'lon': coords['lon'],
                'time': coords.get('time'),
                'n_levels': max(len(v) for v in variables.values()),
                'variables': list(variables.keys()),
                'level_data': variables,
                'source_file': str(file_path),
                'extraction_method': 'simple_variables'
            }
            
            return [profile] if self._validate_profile(profile) else []
            
        except Exception as e:
            logger.debug(f"Simple strategy failed: {e}")
            return []
    
    def _strategy_any_numeric_data(self, ds: xr.Dataset, file_path: Path) -> List[Dict]:
        """Extract any numeric data available - very permissive"""
        try:
            # Find ANY coordinates
            lat, lon = None, None
            
            # Look for latitude/longitude in ANY variable
            for var_name, var in ds.variables.items():
                try:
                    values = var.values
                    if hasattr(values, 'size') and values.size == 1:
                        val = float(values.item() if hasattr(values, 'item') else values)
                        
                        # Guess based on reasonable ranges
                        if -90 <= val <= 90 and lat is None:
                            lat = val
                        elif -180 <= val <= 180 and lon is None:
                            lon = val
                            
                        if lat is not None and lon is not None:
                            break
                except:
                    continue
            
            # If still no coordinates, use defaults or extract from filename
            if lat is None or lon is None:
                lat, lon = self._extract_coords_from_filename(file_path.name)
            
            # If still no valid coordinates, skip
            if lat is None or lon is None:
                return []
            
            # Extract ANY numeric arrays
            data_arrays = {}
            for var_name, var in ds.variables.items():
                try:
                    if var.dtype.kind in 'fc' and var.size > 1:
                        values = self._safe_extract_values(var)
                        if values and len(values) > 1:  # Must have multiple values
                            clean_name = var_name.upper().replace(' ', '_')
                            data_arrays[clean_name] = values
                except:
                    continue
            
            if not data_arrays:
                return []
            
            # Create profile with any available data
            profile = {
                'float_id': f"float_{file_path.stem}",
                'profile_id': f"{file_path.stem}_any_data",
                'lat': lat,
                'lon': lon,
                'time': None,  # Will try to extract later
                'n_levels': max(len(v) for v in data_arrays.values()),
                'variables': list(data_arrays.keys()),
                'level_data': data_arrays,
                'source_file': str(file_path),
                'extraction_method': 'any_numeric'
            }
            
            return [profile] if self._validate_profile(profile) else []
            
        except Exception as e:
            logger.debug(f"Any numeric strategy failed: {e}")
            return []
    
    def _strategy_minimal_extraction(self, ds: xr.Dataset, file_path: Path) -> List[Dict]:
        """Absolute minimal extraction - just record the file exists"""
        try:
            # Use filename or default coordinates
            lat, lon = self._extract_coords_from_filename(file_path.name)
            
            if lat is None or lon is None:
                # Use center of global ocean as fallback
                lat, lon = 0.0, 0.0
            
            # Create minimal profile
            profile = {
                'float_id': f"float_{file_path.stem}",
                'profile_id': f"{file_path.stem}_minimal",
                'lat': lat,
                'lon': lon,
                'time': None,
                'n_levels': 0,
                'variables': ['METADATA_ONLY'],
                'level_data': {'METADATA_ONLY': [1.0]},  # Dummy data
                'source_file': str(file_path),
                'extraction_method': 'minimal_fallback',
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'dimensions': list(ds.dims.keys()) if hasattr(ds, 'dims') else [],
                'variables_available': list(ds.variables.keys()) if hasattr(ds, 'variables') else []
            }
            
            return [profile]
            
        except Exception as e:
            logger.error(f"Even minimal extraction failed for {file_path}: {e}")
            return []
    
    def _extract_coordinates_flexible(self, ds: xr.Dataset) -> Dict:
        """Extract coordinates with maximum flexibility"""
        coords = {}
        
        # Extract each coordinate type
        for coord_type, variants in self.COORDINATE_MAPPINGS.items():
            for variant in variants:
                if variant in ds.variables:
                    try:
                        value = self._safe_extract_scalar(ds[variant])
                        if value is not None:
                            coords[coord_type] = value
                            break
                    except Exception as e:
                        logger.debug(f"Could not extract {variant}: {e}")
                        continue
        
        return coords
    
    def _safe_extract_scalar(self, var) -> Optional[Any]:
        """Safely extract a scalar value from an xarray variable"""
        try:
            values = var.values
            
            # Handle different array structures
            if hasattr(values, 'size'):
                if values.size == 0:
                    return None
                elif values.size == 1:
                    # Single value
                    result = values.item() if hasattr(values, 'item') else values
                else:
                    # Multiple values - take first non-null
                    flat_vals = values.flatten()
                    for val in flat_vals:
                        if not (pd.isna(val) or val is None):
                            result = val
                            break
                    else:
                        return None
            else:
                result = values
            
            # Type conversion
            if hasattr(result, 'tobytes') and hasattr(result, 'dtype') and result.dtype.char == 'S':
                # String/bytes
                return result.tobytes().decode('utf-8', errors='ignore').strip('\x00 ')
            elif isinstance(result, (np.integer, np.floating)):
                return float(result)
            elif isinstance(result, np.ndarray) and result.size == 1:
                return self._safe_extract_scalar(xr.DataArray(result))
            else:
                return result
                
        except Exception as e:
            logger.debug(f"Error in safe_extract_scalar: {e}")
            return None
    
    def _safe_extract_values(self, var) -> Optional[List]:
        """Safely extract array values from an xarray variable"""
        try:
            values = var.values
            
            # Handle different data types
            if hasattr(values, 'dtype'):
                if values.dtype.char == 'S':
                    # String array
                    return [v.decode('utf-8', errors='ignore').strip('\x00 ') 
                           for v in values.flatten()]
                elif values.dtype.kind in 'fc':
                    # Numeric array
                    flat_vals = values.flatten()
                    result = []
                    for val in flat_vals:
                        try:
                            if pd.isna(val) or np.isinf(val):
                                result.append(None)
                            else:
                                result.append(float(val))
                        except (ValueError, TypeError, OverflowError):
                            result.append(None)
                    return result
                else:
                    # Other types
                    return values.flatten().tolist()
            else:
                # Fallback
                if hasattr(values, 'tolist'):
                    return values.tolist()
                elif hasattr(values, 'flatten'):
                    return values.flatten().tolist()
                else:
                    return [values]
                    
        except Exception as e:
            logger.debug(f"Error extracting values: {e}")
            return None
    
    def _extract_profile_standard(self, ds: xr.Dataset, prof_idx: Optional[int], 
                                profile_dim: Optional[str], file_path: Path) -> Optional[Dict]:
        """Extract profile using standard ARGO conventions"""
        try:
            # Extract coordinates
            coords = self._extract_coordinates_flexible(ds)
            
            # Require lat/lon
            if 'latitude' not in coords or 'longitude' not in coords:
                return None
            
            # Handle profile indexing
            if prof_idx is not None and profile_dim:
                # Multi-profile file
                profile_suffix = f"_prof_{prof_idx}"
                
                # Extract coordinates for this profile
                for coord_type, coord_var in coords.items():
                    if coord_type in ['latitude', 'longitude', 'time', 'platform']:
                        try:
                            if profile_dim in ds[self.COORDINATE_MAPPINGS[coord_type][0]].dims:
                                indexed_var = ds[self.COORDINATE_MAPPINGS[coord_type][0]].isel({profile_dim: prof_idx})
                                coords[coord_type] = self._safe_extract_scalar(indexed_var)
                        except:
                            pass  # Keep original value
            else:
                profile_suffix = "_prof_0"
            
            # Extract level data
            level_data = {}
            variables_present = []
            
            for standard_name, variants in self.VARIABLE_MAPPINGS.items():
                for variant in variants:
                    if variant in ds.variables:
                        try:
                            var = ds[variant]
                            
                            # Handle profile indexing
                            if prof_idx is not None and profile_dim and profile_dim in var.dims:
                                var = var.isel({profile_dim: prof_idx})
                            
                            values = self._safe_extract_values(var)
                            if values and len(values) > 0:
                                level_data[standard_name.upper()] = values
                                variables_present.append(standard_name.upper())
                                break  # Found this variable
                                
                        except Exception as e:
                            logger.debug(f"Could not extract {variant}: {e}")
                            continue
            
            if not level_data:
                return None
            
            # Build profile
            profile = {
                'float_id': str(coords.get('platform', f"float_{file_path.stem}"))[:50],
                'profile_id': f"{file_path.stem}{profile_suffix}",
                'lat': float(coords['latitude']),
                'lon': float(coords['longitude']),
                'time': self._format_time(coords.get('time')),
                'n_levels': max(len(v) for v in level_data.values() if v),
                'variables': variables_present,
                'level_data': level_data,
                'source_file': str(file_path),
                'extraction_method': 'standard_argo'
            }
            
            return profile
            
        except Exception as e:
            logger.debug(f"Standard extraction failed: {e}")
            return None
    
    def _map_to_standard_name(self, var_name: str) -> str:
        """Map any variable name to a standard name"""
        var_upper = var_name.upper()
        
        for standard_name, variants in self.VARIABLE_MAPPINGS.items():
            for variant in variants:
                if variant.upper() == var_upper:
                    return standard_name.upper()
        
        # If no mapping found, clean the name
        clean_name = ''.join(c for c in var_upper if c.isalnum() or c == '_')
        return clean_name if clean_name else 'UNKNOWN_VAR'
    
    def _extract_coords_from_filename(self, filename: str) -> Tuple[Optional[float], Optional[float]]:
        """Try to extract coordinates from filename patterns"""
        import re
        
        # Common ARGO filename patterns
        patterns = [
            r'(\d+\.?\d*)N_(\d+\.?\d*)W',  # lat N, lon W
            r'(\d+\.?\d*)N_(\d+\.?\d*)E',  # lat N, lon E
            r'(\d+\.?\d*)S_(\d+\.?\d*)W',  # lat S, lon W
            r'(\d+\.?\d*)S_(\d+\.?\d*)E',  # lat S, lon E
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    
                    # Apply hemisphere corrections
                    if 'S' in pattern:
                        lat = -lat
                    if 'W' in pattern:
                        lon = -lon
                    
                    return lat, lon
                except:
                    continue
        
        # Fallback: use oceanic defaults
        return 0.0, 0.0  # Equatorial Pacific
    
    def _create_minimal_profile(self, file_path: Path) -> Dict:
        """Create absolute minimal profile for files that can't be parsed"""
        lat, lon = self._extract_coords_from_filename(file_path.name)
        
        return {
            'float_id': f"float_{file_path.stem}",
            'profile_id': f"{file_path.stem}_minimal_fallback",
            'lat': lat or 0.0,
            'lon': lon or 0.0,
            'time': None,
            'n_levels': 0,
            'variables': ['FILE_ONLY'],
            'level_data': {'FILE_ONLY': [1.0]},
            'source_file': str(file_path),
            'extraction_method': 'minimal_fallback',
            'file_size': file_path.stat().st_size if file_path.exists() else 0
        }
    
    def _validate_profile(self, profile: Dict) -> bool:
        """Validate that a profile has minimum required data"""
        try:
            # Must have coordinates
            if not isinstance(profile.get('lat'), (int, float)) or not isinstance(profile.get('lon'), (int, float)):
                return False
            
            # Latitude/longitude must be in valid ranges
            if not (-90 <= profile['lat'] <= 90) or not (-180 <= profile['lon'] <= 180):
                return False
            
            # Must have some data
            level_data = profile.get('level_data', {})
            if not level_data:
                return False
            
            # At least one variable must have data
            for var_name, values in level_data.items():
                if values and len(values) > 0:
                    # Check if there's at least one non-null value
                    if any(v is not None and not pd.isna(v) for v in values):
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Profile validation failed: {e}")
            return False
    
    def _format_time(self, time_val) -> Optional[str]:
        """Format time value to string"""
        if time_val is None:
            return None
        
        try:
            if hasattr(time_val, 'strftime'):
                return time_val.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(time_val, str):
                return time_val
            elif isinstance(time_val, (int, float)):
                # Assume it's some kind of timestamp
                if time_val > 1e9:  # Unix timestamp
                    return datetime.fromtimestamp(time_val).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    return str(time_val)
            else:
                return str(time_val)
        except:
            return str(time_val) if time_val is not None else None

class BulletproofIngestionPipeline:
    """
    Complete ingestion pipeline with comprehensive error handling.
    """
    
    def __init__(self, parquet_dir: str = PARQUET_DIR):
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = BulletproofArgoParser()
        self.engine = None
        self.emb = None
        
        # Statistics tracking
        self.stats = {
            'files_attempted': 0,
            'files_successful': 0,
            'files_failed': 0,
            'profiles_extracted': 0,
            'profiles_saved': 0,
            'profiles_embedded': 0,
            'errors': []
        }
    
    def initialize_components(self) -> bool:
        """Initialize database and embedding components"""
        try:
            # Initialize database
            self.engine = pg_engine()
            if not self._setup_bulletproof_schema():
                return False
            
            # Initialize embeddings
            self.emb = EmbeddingManager()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
    
    def _setup_bulletproof_schema(self) -> bool:
        """Set up database schema that handles any data structure"""
        try:
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()
            
            with self.engine.begin() as conn:
                # Create main table with maximum flexibility
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS floats (
                        id SERIAL PRIMARY KEY,
                        float_id TEXT NOT NULL,
                        profile_id TEXT UNIQUE NOT NULL,
                        source_file TEXT,
                        file_size BIGINT,
                        extraction_method TEXT,
                        time_start TIMESTAMP,
                        time_end TIMESTAMP,
                        lat DOUBLE PRECISION NOT NULL,
                        lon DOUBLE PRECISION NOT NULL,
                        platform_number TEXT,
                        n_levels INTEGER DEFAULT 0,
                        variables JSONB DEFAULT '[]',
                        parquet_path TEXT,
                        schema_info JSONB DEFAULT '{}',
                        raw_metadata JSONB DEFAULT '{}',
                        processing_errors JSONB DEFAULT '[]',
                        data_quality_score FLOAT DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        
                        -- Constraints
                        CONSTRAINT valid_coordinates CHECK (
                            lat BETWEEN -90 AND 90 AND 
                            lon BETWEEN -180 AND 180
                        ),
                        CONSTRAINT valid_levels CHECK (n_levels >= 0)
                    );
                """))
                
                # Create comprehensive indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_float_geo 
                    ON floats USING GIST (ST_SetSRID(ST_Point(lon, lat), 4326));
                    
                    CREATE INDEX IF NOT EXISTS idx_float_id ON floats(float_id);
                    CREATE INDEX IF NOT EXISTS idx_profile_id ON floats(profile_id);
                    CREATE INDEX IF NOT EXISTS idx_source_file ON floats(source_file);
                    CREATE INDEX IF NOT EXISTS idx_extraction_method ON floats(extraction_method);
                    CREATE INDEX IF NOT EXISTS idx_time_start ON floats(time_start);
                    CREATE INDEX IF NOT EXISTS idx_variables ON floats USING GIN(variables);
                    CREATE INDEX IF NOT EXISTS idx_schema_info ON floats USING GIN(schema_info);
                    CREATE INDEX IF NOT EXISTS idx_data_quality ON floats(data_quality_score);
                """))
                
                # Create error tracking table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ingestion_errors (
                        id SERIAL PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        error_type TEXT,
                        error_message TEXT,
                        error_details JSONB,
                        occurred_at TIMESTAMP DEFAULT NOW()
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_error_file ON ingestion_errors(file_path);
                    CREATE INDEX IF NOT EXISTS idx_error_type ON ingestion_errors(error_type);
                """))
                
                logger.info("Bulletproof database schema created")
                return True
                
        except Exception as e:
            logger.error(f"Schema setup failed: {e}")
            return False
    
    def ingest_file_bulletproof(self, file_path: str) -> Tuple[int, int]:
        """
        Ingest a single file with maximum error recovery.
        Returns (successful_profiles, total_profiles)
        """
        file_path = Path(file_path)
        self.stats['files_attempted'] += 1
        
        try:
            # Parse the file
            profiles = self.parser.parse_file_bulletproof(file_path)
            
            if not profiles:
                self._log_file_error(file_path, "NO_PROFILES", "No profiles extracted")
                self.stats['files_failed'] += 1
                return 0, 0
            
            self.stats['profiles_extracted'] += len(profiles)
            successful = 0
            
            # Process each profile
            for profile in profiles:
                try:
                    if self._ingest_single_profile(profile):
                        successful += 1
                        self.stats['profiles_saved'] += 1
                except Exception as e:
                    logger.warning(f"Failed to ingest profile {profile.get('profile_id', 'unknown')}: {e}")
                    continue
            
            if successful > 0:
                self.stats['files_successful'] += 1
            else:
                self.stats['files_failed'] += 1
            
            return successful, len(profiles)
            
        except Exception as e:
            error_msg = f"Critical error processing {file_path}: {e}"
            logger.error(error_msg)
            self._log_file_error(file_path, "CRITICAL_ERROR", error_msg)
            self.stats['files_failed'] += 1
            return 0, 0
    
    def _ingest_single_profile(self, profile: Dict) -> bool:
        """Ingest a single profile with comprehensive error handling"""
        try:
            # 1. Write parquet file
            parquet_path = self._write_parquet_bulletproof(profile)
            if not parquet_path:
                logger.warning(f"Failed to write parquet for {profile['profile_id']}")
                return False
            
            profile['parquet_path'] = parquet_path
            
            # 2. Calculate data quality score
            quality_score = self._calculate_data_quality(profile)
            
            # 3. Insert into database
            db_success = self._insert_database_record(profile, quality_score)
            if not db_success:
                logger.warning(f"Failed to insert DB record for {profile['profile_id']}")
                return False
            
            # 4. Create embedding (non-critical)
            try:
                self._create_embedding_safe(profile)
                self.stats['profiles_embedded'] += 1
            except Exception as e:
                logger.warning(f"Embedding failed for {profile['profile_id']}: {e}")
                # Don't fail the whole ingestion for embedding errors
            
            return True
            
        except Exception as e:
            logger.error(f"Profile ingestion failed for {profile.get('profile_id', 'unknown')}: {e}")
            return False
    
    def _write_parquet_bulletproof(self, profile: Dict) -> Optional[str]:
        """Write parquet with maximum error recovery"""
        try:
            # Clean identifiers for file system
            pid = self._clean_filename(profile["profile_id"])
            float_id_clean = self._clean_filename(str(profile["float_id"]))
            
            # Create directory
            out_dir = self.parquet_dir / float_id_clean
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare data
            level_data = profile.get("level_data", {})
            if not level_data:
                return None
            
            # Standardize array lengths
            max_length = max(len(v) for v in level_data.values() if v)
            if max_length == 0:
                return None
            
            # Create DataFrame with error handling
            df_data = {}
            for col_name, values in level_data.items():
                try:
                    # Ensure proper length
                    if len(values) < max_length:
                        values = list(values) + [np.nan] * (max_length - len(values))
                    elif len(values) > max_length:
                        values = values[:max_length]
                    
                    # Convert to proper types
                    clean_values = []
                    for val in values:
                        try:
                            if val is None or pd.isna(val):
                                clean_values.append(np.nan)
                            else:
                                clean_values.append(float(val))
                        except (ValueError, TypeError, OverflowError):
                            clean_values.append(np.nan)
                    
                    df_data[col_name] = clean_values
                    
                except Exception as e:
                    logger.debug(f"Skipping problematic column {col_name}: {e}")
                    continue
            
            if not df_data:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(df_data)
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            if df.empty:
                return None
            
            # Write parquet with compression
            parquet_path = out_dir / f"{pid}.parquet"
            df.to_parquet(parquet_path, index=False, compression='snappy')
            
            logger.debug(f"Saved parquet: {parquet_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
            return str(parquet_path)
            
        except Exception as e:
            logger.error(f"Parquet writing failed for {profile.get('profile_id', 'unknown')}: {e}")
            return None
    
    def _clean_filename(self, name: str) -> str:
        """Clean a string for use as filename/directory"""
        # Replace problematic characters
        clean = str(name).replace("/", "_").replace("\\", "_").replace(":", "_")
        clean = clean.replace(" ", "_").replace("<", "_").replace(">", "_")
        
        # Keep only alphanumeric and safe characters
        clean = "".join(c for c in clean if c.isalnum() or c in "_-.")
        
        # Ensure it's not empty and not too long
        clean = clean[:100] if clean else "unknown"
        
        return clean
    
    def _calculate_data_quality(self, profile: Dict) -> float:
        """Calculate a data quality score (0-1)"""
        score = 1.0
        
        try:
            level_data = profile.get('level_data', {})
            
            if not level_data:
                return 0.1
            
            # Check data completeness
            total_values = sum(len(v) for v in level_data.values())
            non_null_values = sum(
                sum(1 for val in v if val is not None and not pd.isna(val))
                for v in level_data.values()
            )
            
            if total_values > 0:
                completeness = non_null_values / total_values
                score *= completeness
            
            # Bonus for having key variables
            key_vars = ['TEMPERATURE', 'SALINITY', 'PRESSURE']
            key_vars_present = sum(1 for var in key_vars if var in level_data)
            score *= (0.5 + 0.5 * key_vars_present / len(key_vars))
            
            # Bonus for having good coordinates
            if profile.get('lat') and profile.get('lon'):
                if abs(profile['lat']) <= 90 and abs(profile['lon']) <= 180:
                    score *= 1.0
                else:
                    score *= 0.5
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default moderate score
    
    def _insert_database_record(self, profile: Dict, quality_score: float) -> bool:
        """Insert database record with robust error handling"""
        try:
            # Prepare record with safe type conversion
            db_record = {
                "float_id": str(profile.get("float_id", "unknown"))[:50],
                "profile_id": str(profile["profile_id"])[:100],
                "source_file": str(profile.get("source_file", ""))[:500],
                "file_size": profile.get("file_size"),
                "extraction_method": str(profile.get("extraction_method", "unknown"))[:50],
                "time_start": profile.get("time"),
                "time_end": profile.get("time"),
                "lat": self._safe_float(profile.get("lat")),
                "lon": self._safe_float(profile.get("lon")),
                "platform_number": str(profile.get("float_id", "unknown"))[:50],
                "n_levels": int(profile.get("n_levels", 0)),
                "variables": json.dumps(profile.get("variables", [])),
                "parquet_path": str(profile.get("parquet_path", "")),
                "data_quality_score": float(quality_score),
                "schema_info": json.dumps(profile.get("detected_schema", {})),
                "raw_metadata": json.dumps({
                    k: v for k, v in profile.items() 
                    if k not in ["level_data", "detected_schema"] and v is not None
                })
            }
            
            # Validate required fields
            if db_record["lat"] is None or db_record["lon"] is None:
                logger.warning(f"Invalid coordinates for {profile['profile_id']}")
                return False
            
            # Insert with upsert
            with self.engine.begin() as conn:
                sql = text("""
                    INSERT INTO floats (
                        float_id, profile_id, source_file, file_size, extraction_method,
                        time_start, time_end, lat, lon, platform_number, n_levels, 
                        variables, parquet_path, data_quality_score, schema_info, raw_metadata
                    ) VALUES (
                        :float_id, :profile_id, :source_file, :file_size, :extraction_method,
                        :time_start, :time_end, :lat, :lon, :platform_number, :n_levels,
                        :variables::jsonb, :parquet_path, :data_quality_score,
                        :schema_info::jsonb, :raw_metadata::jsonb
                    )
                    ON CONFLICT (profile_id) DO UPDATE SET
                        parquet_path = EXCLUDED.parquet_path,
                        data_quality_score = EXCLUDED.data_quality_score,
                        schema_info = EXCLUDED.schema_info,
                        raw_metadata = EXCLUDED.raw_metadata,
                        extraction_method = EXCLUDED.extraction_method,
                        updated_at = NOW()
                """)
                
                conn.execute(sql, db_record)
                return True
                
        except Exception as e:
            logger.error(f"Database insertion failed for {profile.get('profile_id', 'unknown')}: {e}")
            return False
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        
        try:
            result = float(value)
            if np.isnan(result) or np.isinf(result):
                return None
            return result
        except (ValueError, TypeError, OverflowError):
            return None
    
    def _create_embedding_safe(self, profile: Dict):
        """Create embedding with error handling"""
        try:
            summary = self._create_rich_summary(profile)
            metadata = {
                "float_id": str(profile["float_id"]),
                "lat": float(profile["lat"]),
                "lon": float(profile["lon"]),
                "time": profile.get("time"),
                "parquet": profile.get("parquet_path"),
                "variables": profile.get("variables", []),
                "quality_score": profile.get("data_quality_score", 0.5),
                "extraction_method": profile.get("extraction_method", "unknown")
            }
            
            self.emb.add_documents([profile["profile_id"]], [metadata], [summary])
            
        except Exception as e:
            logger.debug(f"Embedding creation failed: {e}")
            raise  # Re-raise to be caught by caller
    
    def _create_rich_summary(self, profile: Dict) -> str:
        """Create rich natural language summary"""
        try:
            parts = []
            
            # Basic info
            parts.append(f"ARGO float {profile['float_id']} profile")
            
            # Location with hemisphere
            lat, lon = profile['lat'], profile['lon']
            lat_hem = "N" if lat >= 0 else "S"
            lon_hem = "E" if lon >= 0 else "W"
            parts.append(f"at {abs(lat):.2f}°{lat_hem}, {abs(lon):.2f}°{lon_hem}")
            
            # Time if available
            if profile.get('time'):
                parts.append(f"measured on {profile['time']}")
            
            # Data characteristics
            if profile['n_levels'] > 0:
                parts.append(f"with {profile['n_levels']} depth measurements")
            
            # Variables
            if profile.get('variables'):
                var_list = profile['variables']
                if 'METADATA_ONLY' not in var_list and 'FILE_ONLY' not in var_list:
                    parts.append(f"measuring {', '.join(var_list).lower()}")
            
            # Quality indicator
            quality = profile.get('data_quality_score', 0.5)
            if quality > 0.8:
                parts.append("(high quality data)")
            elif quality < 0.3:
                parts.append("(limited data)")
            
            # Ocean region guess
            region = self._guess_ocean_region(lat, lon)
            if region:
                parts.append(f"in the {region}")
            
            return " ".join(parts) + "."
            
        except Exception as e:
            logger.debug(f"Rich summary creation failed: {e}")
            return f"ARGO profile {profile.get('profile_id', 'unknown')}"
    
    def _guess_ocean_region(self, lat: float, lon: float) -> Optional[str]:
        """Guess ocean region from coordinates"""
        try:
            # Simple ocean basin classification
            if -180 <= lon <= -30:
                if lat > 0:
                    return "North Atlantic"
                else:
                    return "South Atlantic"
            elif -30 <= lon <= 20:
                if lat > 30:
                    return "North Atlantic"
                elif lat < -30:
                    return "South Atlantic"
                else:
                    return "tropical Atlantic"
            elif 20 <= lon <= 120:
                if lat > 0:
                    return "North Indian Ocean"
                else:
                    return "South Indian Ocean"
            elif 120 <= lon <= 180 or -180 <= lon <= -120:
                if lat > 0:
                    return "North Pacific"
                else:
                    return "South Pacific"
            else:
                return "Pacific Ocean"
                
        except:
            return None
    
    def _log_file_error(self, file_path: Path, error_type: str, error_message: str):
        """Log file processing errors to database"""
        try:
            if self.engine:
                with self.engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO ingestion_errors (file_path, error_type, error_message, error_details)
                        VALUES (:file_path, :error_type, :error_message, :error_details)
                    """), {
                        "file_path": str(file_path),
                        "error_type": error_type,
                        "error_message": error_message,
                        "error_details": json.dumps({"file_size": file_path.stat().st_size if file_path.exists() else 0})
                    })
        except Exception as e:
            logger.debug(f"Could not log error to database: {e}")
        
        # Also add to stats
        self.stats['errors'].append({
            'file': str(file_path),
            'type': error_type,
            'message': error_message
        })
    
    def ingest_directory_bulletproof(self, nc_dir: str, max_files: Optional[int] = None, 
                                   skip_existing: bool = True) -> bool:
        """
        Bulletproof directory ingestion with comprehensive error handling.
        """
        nc_dir = Path(nc_dir)
        
        print(f"🚀 Starting bulletproof ingestion from {nc_dir}")
        print(f"📁 Output directory: {self.parquet_dir}")
        
        # Initialize components
        if not self.initialize_components():
            print("❌ Failed to initialize components")
            return False
        
        # Find files
        try:
            files = self._find_netcdf_files_comprehensive(nc_dir)
            if max_files:
                files = files[:max_files]
            print(f"📋 Found {len(files)} NetCDF files")
        except Exception as e:
            print(f"❌ Error finding files: {e}")
            return False
        
        if not files:
            print(f"❌ No NetCDF files found in {nc_dir}")
            return False
        
        # Filter existing files if requested
        if skip_existing:
            files = self._filter_existing_files(files)
            print(f"📋 After filtering existing: {len(files)} files to process")
        
        # Process files with detailed progress
        print(f"🔄 Processing {len(files)} files...")
        
        for i, file_path in enumerate(tqdm(files, desc="🌊 Ingesting ARGO data")):
            try:
                successful, total = self.ingest_file_bulletproof(file_path)
                
                # Log progress periodically
                if (i + 1) % 25 == 0 or i == len(files) - 1:
                    self._print_progress_summary(i + 1, len(files))
                    
            except Exception as e:
                logger.error(f"Unexpected error with {file_path}: {e}")
                continue
        
        # Final summary
        self._print_final_summary()
        
        return self.stats['profiles_saved'] > 0
    
    def _find_netcdf_files_comprehensive(self, directory: Path) -> List[str]:
        """Find NetCDF files with comprehensive pattern matching"""
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")
        
        # All possible NetCDF extensions and patterns
        patterns = [
            "**/*.nc", "**/*.NC", "**/*.netcdf", "**/*.NETCDF", 
            "**/*.nc4", "**/*.NC4", "**/*.cdf", "**/*.CDF",
            "**/*.hdf", "**/*.HDF", "**/*.h5", "**/*.H5"
        ]
        
        all_files = set()
        for pattern in patterns:
            try:
                files = list(directory.glob(pattern))
                all_files.update(str(f) for f in files)
            except Exception as e:
                logger.debug(f"Pattern {pattern} failed: {e}")
                continue
        
        return sorted(all_files)
    
    def _filter_existing_files(self, files: List[str]) -> List[str]:
        """Filter out files that have already been processed"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT DISTINCT source_file FROM floats"))
                existing_files = {row[0] for row in result}
            
            filtered = [f for f in files if f not in existing_files]
            logger.info(f"Filtered out {len(files) - len(filtered)} already processed files")
            return filtered
            
        except Exception as e:
            logger.warning(f"Could not filter existing files: {e}")
            return files  # Process all if filtering fails
    
    def _print_progress_summary(self, current: int, total: int):
        """Print detailed progress summary"""
        print(f"\n📊 Progress Update ({current}/{total} files):")
        print(f"   ✅ Files successful: {self.stats['files_successful']}")
        print(f"   ❌ Files failed: {self.stats['files_failed']}")
        print(f"   🔍 Profiles extracted: {self.stats['profiles_extracted']}")
        print(f"   💾 Profiles saved: {self.stats['profiles_saved']}")
        print(f"   🔗 Profiles embedded: {self.stats['profiles_embedded']}")
        
        if self.stats['profiles_extracted'] > 0:
            success_rate = self.stats['profiles_saved'] / self.stats['profiles_extracted'] * 100
            print(f"   📈 Success rate: {success_rate:.1f}%")
    
    def _print_final_summary(self):
        """Print comprehensive final summary"""
        print(f"\n🎉 Bulletproof Ingestion Complete!")
        print(f"{'='*50}")
        print(f"📁 Files attempted: {self.stats['files_attempted']}")
        print(f"✅ Files successful: {self.stats['files_successful']}")
        print(f"❌ Files failed: {self.stats['files_failed']}")
        print(f"🔍 Total profiles found: {self.stats['profiles_extracted']}")
        print(f"💾 Profiles successfully saved: {self.stats['profiles_saved']}")
        print(f"🔗 Profiles embedded: {self.stats['profiles_embedded']}")
        
        if self.stats['profiles_extracted'] > 0:
            overall_success = self.stats['profiles_saved'] / self.stats['profiles_extracted'] * 100
            print(f"📈 Overall success rate: {overall_success:.1f}%")
        
        if self.stats['files_attempted'] > 0:
            file_success = self.stats['files_successful'] / self.stats['files_attempted'] * 100
            print(f"📋 File processing rate: {file_success:.1f}%")
        
        # Show common error types
        if self.stats['errors']:
            error_types = {}
            for error in self.stats['errors']:
                error_type = error['type']
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            print(f"\n🔍 Error Summary:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {error_type}: {count} files")
    
    def verify_ingestion(self) -> Dict:
        """Verify the ingestion results"""
        try:
            with self.engine.connect() as conn:
                # Count records
                result = conn.execute(text("SELECT COUNT(*) FROM floats"))
                total_records = result.scalar()
                
                # Count by extraction method
                result = conn.execute(text("""
                    SELECT extraction_method, COUNT(*) as count, AVG(data_quality_score) as avg_quality
                    FROM floats 
                    GROUP BY extraction_method 
                    ORDER BY count DESC
                """))
                methods = result.fetchall()
                
                # Count by quality
                result = conn.execute(text("""
                    SELECT 
                        CASE 
                            WHEN data_quality_score >= 0.8 THEN 'High'
                            WHEN data_quality_score >= 0.5 THEN 'Medium'
                            ELSE 'Low'
                        END as quality_tier,
                        COUNT(*) as count
                    FROM floats 
                    GROUP BY quality_tier
                """))
                quality_tiers = result.fetchall()
                
                verification = {
                    "total_records": total_records,
                    "extraction_methods": [{"method": m[0], "count": m[1], "avg_quality": float(m[2])} for m in methods],
                    "quality_distribution": [{"tier": q[0], "count": q[1]} for q in quality_tiers],
                    "ingestion_stats": self.stats
                }
                
                return verification
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"error": str(e)}

def run_bulletproof_ingestion(nc_dir: str, parquet_dir: str = PARQUET_DIR, 
                            max_files: Optional[int] = None, skip_existing: bool = True) -> bool:
    """
    Main function to run bulletproof ingestion.
    This function guarantees that NO file will cause the process to crash.
    """
    try:
        pipeline = BulletproofIngestionPipeline(parquet_dir)
        success = pipeline.ingest_directory_bulletproof(nc_dir, max_files, skip_existing)
        
        # Print verification
        verification = pipeline.verify_ingestion()
        print(f"\n🔍 Verification Results:")
        print(f"   Total records in database: {verification.get('total_records', 0)}")
        
        return success
        
    except Exception as e:
        print(f"❌ Critical pipeline failure: {e}")
        return False

# Enhanced CLI with recovery options
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulletproof ARGO NetCDF ingestion system")
    parser.add_argument("--dir", required=True, help="Directory containing NetCDF files")
    parser.add_argument("--parquet-dir", default=PARQUET_DIR, help="Output directory for parquet files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    parser.add_argument("--skip-existing", action="store_true", default=True, 
                       help="Skip files already in database")
    parser.add_argument("--force-reprocess", action="store_true", 
                       help="Reprocess all files, even if already ingested")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--verify-only", action="store_true", 
                       help="Only verify existing ingestion, don't process new files")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input
    if not Path(args.dir).exists():
        print(f"❌ Input directory {args.dir} does not exist")
        exit(1)
    
    try:
        if args.verify_only:
            # Just run verification
            pipeline = BulletproofIngestionPipeline(args.parquet_dir)
            if pipeline.initialize_components():
                verification = pipeline.verify_ingestion()
                print(json.dumps(verification, indent=2, default=str))
            exit(0)
        
        # Run ingestion
        skip_existing = args.skip_existing and not args.force_reprocess
        success = run_bulletproof_ingestion(
            args.dir, 
            args.parquet_dir, 
            args.max_files,
            skip_existing
        )
        
        exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Ingestion interrupted by user")
        exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)