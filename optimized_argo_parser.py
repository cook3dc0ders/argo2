# optimized_argo_parser.py
"""
Optimized ARGO parser specifically tuned for your schema:
N_LEVELS,TEMP,PSAL,DOXY,PRES,PLATFORM_NUMBER,LATITUDE,LONGITUDE,JULD

This parser is optimized for speed and reliability with your exact column structure.
"""
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class OptimizedArgoParser:
    """
    Ultra-fast parser optimized for your exact ARGO schema.
    Handles your specific column names with maximum efficiency.
    """
    
    # Your exact schema mapping
    SCHEMA_MAP = {
        'temperature': 'TEMP',
        'salinity': 'PSAL', 
        'oxygen': 'DOXY',
        'pressure': 'PRES',
        'latitude': 'LATITUDE',
        'longitude': 'LONGITUDE',
        'time': 'JULD',
        'platform': 'PLATFORM_NUMBER',
        'n_levels': 'N_LEVELS'
    }
    
    def __init__(self):
        self.files_processed = 0
        self.profiles_extracted = 0
    
    def parse_your_netcdf(self, file_path: str) -> List[Dict]:
        """
        Parse NetCDF file with your exact schema.
        Optimized for maximum speed and reliability.
        """
        profiles = []
        file_path = Path(file_path)
        
        try:
            # Open file with optimized settings for your schema
            ds = xr.open_dataset(file_path, decode_times=False, mask_and_scale=True)
            
            # Quick validation - check if our expected variables exist
            required_vars = ['TEMP', 'PSAL', 'LATITUDE', 'LONGITUDE', 'PLATFORM_NUMBER']
            missing_vars = [var for var in required_vars if var not in ds.variables]
            
            if missing_vars:
                logger.warning(f"Missing variables in {file_path.name}: {missing_vars}")
                # Fall back to bulletproof parser for unusual files
                return self._fallback_parse(ds, file_path)
            
            # Detect profile structure
            profile_count = self._detect_profile_count(ds)
            
            # Extract profiles
            for prof_idx in range(profile_count):
                profile = self._extract_optimized_profile(ds, prof_idx, profile_count, file_path)
                if profile and self._validate_profile(profile):
                    profiles.append(profile)
                    self.profiles_extracted += 1
            
            self.files_processed += 1
            logger.info(f"✅ Processed {file_path.name}: {len(profiles)} profiles")
            
        except Exception as e:
            logger.warning(f"Optimized parsing failed for {file_path.name}: {e}")
            # Fallback to bulletproof method
            try:
                ds = xr.open_dataset(file_path, decode_times=False, mask_and_scale=True)
                profiles = self._fallback_parse(ds, file_path)
            except Exception as e2:
                logger.error(f"Both optimized and fallback parsing failed for {file_path.name}: {e2}")
                profiles = []
        
        finally:
            try:
                if 'ds' in locals():
                    ds.close()
            except:
                pass
        
        return profiles
    
    def _detect_profile_count(self, ds: xr.Dataset) -> int:
        """Detect how many profiles are in the dataset"""
        try:
            # Check for profile dimension
            profile_dims = ['N_PROF', 'profile', 'PROFILE']
            for dim in profile_dims:
                if dim in ds.dims:
                    return ds.sizes[dim]
            
            # If no explicit profile dimension, check data shapes
            # Most variables should have the same leading dimension
            if 'TEMP' in ds.variables:
                temp_dims = ds['TEMP'].dims
                if len(temp_dims) >= 2:
                    # Assume first dimension is profiles
                    return ds.sizes[temp_dims[0]]
            
            # Default to single profile
            return 1
            
        except Exception as e:
            logger.debug(f"Profile count detection failed: {e}")
            return 1
    
    def _extract_optimized_profile(self, ds: xr.Dataset, prof_idx: int, 
                                 total_profiles: int, file_path: Path) -> Optional[Dict]:
        """Extract single profile using your exact schema"""
        try:
            # Extract coordinates (scalars or profile-specific)
            lat = self._extract_coordinate(ds, 'LATITUDE', prof_idx, total_profiles)
            lon = self._extract_coordinate(ds, 'LONGITUDE', prof_idx, total_profiles)
            time_val = self._extract_coordinate(ds, 'JULD', prof_idx, total_profiles)
            platform = self._extract_coordinate(ds, 'PLATFORM_NUMBER', prof_idx, total_profiles)
            
            # Validate essential coordinates
            if lat is None or lon is None:
                logger.warning(f"Missing coordinates for profile {prof_idx} in {file_path.name}")
                return None
            
            # Ensure coordinates are valid
            try:
                lat = float(lat)
                lon = float(lon)
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
                    return None
            except (ValueError, TypeError):
                logger.warning(f"Cannot convert coordinates to float: lat={lat}, lon={lon}")
                return None
            
            # Extract level data arrays
            level_data = {}
            variables_present = []
            
            # Temperature
            temp = self._extract_level_data(ds, 'TEMP', prof_idx, total_profiles)
            if temp:
                level_data['TEMPERATURE'] = temp
                variables_present.append('TEMPERATURE')
            
            # Salinity  
            sal = self._extract_level_data(ds, 'PSAL', prof_idx, total_profiles)
            if sal:
                level_data['SALINITY'] = sal
                variables_present.append('SALINITY')
            
            # Oxygen
            oxy = self._extract_level_data(ds, 'DOXY', prof_idx, total_profiles)
            if oxy:
                level_data['OXYGEN'] = oxy
                variables_present.append('OXYGEN')
            
            # Pressure
            pres = self._extract_level_data(ds, 'PRES', prof_idx, total_profiles)
            if pres:
                level_data['PRESSURE'] = pres
                variables_present.append('PRESSURE')
            
            # Check if we have any actual data
            if not level_data:
                logger.warning(f"No level data found for profile {prof_idx}")
                return None
            
            # Calculate number of levels
            n_levels = max(len(v) for v in level_data.values() if v)
            
            # Create profile ID
            profile_suffix = f"_prof_{prof_idx}" if total_profiles > 1 else "_prof_0"
            profile_id = f"{file_path.stem}{profile_suffix}"
            
            # Format platform number
            platform_str = str(platform) if platform else f"float_{file_path.stem}"
            
            # Format time
            time_str = self._format_time(time_val)
            
            profile = {
                'float_id': platform_str[:50],
                'profile_id': profile_id,
                'lat': lat,
                'lon': lon,
                'time': time_str,
                'n_levels': n_levels,
                'variables': variables_present,
                'level_data': level_data,
                'source_file': str(file_path),
                'extraction_method': 'optimized_schema',
                'schema_version': 'your_standard_schema'
            }
            
            return profile
            
        except Exception as e:
            logger.error(f"Error extracting profile {prof_idx} from {file_path.name}: {e}")
            return None
    
    def _extract_coordinate(self, ds: xr.Dataset, var_name: str, 
                          prof_idx: int, total_profiles: int) -> Any:
        """Extract coordinate value (scalar or profile-indexed)"""
        if var_name not in ds.variables:
            return None
        
        try:
            var = ds[var_name]
            
            # Handle different dimensionalities
            if var.ndim == 0:
                # Scalar
                value = var.item()
            elif var.ndim == 1:
                # Vector - could be profiles or something else
                if total_profiles > 1 and len(var) == total_profiles:
                    # Profile dimension
                    value = var.isel({var.dims[0]: prof_idx}).item()
                else:
                    # Take first value
                    value = var.isel({var.dims[0]: 0}).item()
            else:
                # Multi-dimensional - take first available value
                # Flatten and take first non-null
                flat_vals = var.values.flatten()
                for val in flat_vals:
                    if not pd.isna(val):
                        value = val
                        break
                else:
                    value = None
            
            # Clean string values
            if hasattr(value, 'tobytes'):
                value = value.tobytes().decode('utf-8', errors='ignore').strip('\x00 ')
            elif isinstance(value, bytes):
                value = value.decode('utf-8', errors='ignore').strip('\x00 ')
            
            return value
            
        except Exception as e:
            logger.debug(f"Error extracting {var_name}: {e}")
            return None
    
    def _extract_level_data(self, ds: xr.Dataset, var_name: str, 
                           prof_idx: int, total_profiles: int) -> Optional[List]:
        """Extract array of level data"""
        if var_name not in ds.variables:
            return None
        
        try:
            var = ds[var_name]
            
            # Handle different dimensionalities
            if var.ndim == 1:
                # Simple 1D array
                values = var.values
            elif var.ndim == 2:
                # 2D array - assume [profiles, levels] or [levels, profiles]
                if total_profiles > 1:
                    # Multi-profile case
                    if var.shape[0] == total_profiles:
                        # [profiles, levels]
                        values = var.isel({var.dims[0]: prof_idx}).values
                    elif var.shape[1] == total_profiles:
                        # [levels, profiles]  
                        values = var.isel({var.dims[1]: prof_idx}).values
                    else:
                        # Take first profile's worth of data
                        values = var.isel({var.dims[0]: 0}).values
                else:
                    # Single profile - take all data from appropriate dimension
                    # Choose the longer dimension as levels
                    if var.shape[0] >= var.shape[1]:
                        values = var.isel({var.dims[1]: 0}).values
                    else:
                        values = var.isel({var.dims[0]: 0}).values
            else:
                # Higher dimensional - flatten and take what we can
                values = var.values.flatten()
            
            # Convert to clean list
            clean_values = []
            for val in values:
                try:
                    if pd.isna(val) or np.isinf(val):
                        clean_values.append(None)
                    else:
                        clean_values.append(float(val))
                except (ValueError, TypeError, OverflowError):
                    clean_values.append(None)
            
            # Remove trailing None values but keep structure
            while clean_values and clean_values[-1] is None:
                clean_values.pop()
            
            return clean_values if clean_values else None
            
        except Exception as e:
            logger.debug(f"Error extracting level data for {var_name}: {e}")
            return None
    
    def _format_time(self, time_val: Any) -> Optional[str]:
        """Format time value to string"""
        if time_val is None:
            return None
        
        try:
            # Handle different time formats
            if isinstance(time_val, str):
                return time_val
            elif hasattr(time_val, 'strftime'):
                return time_val.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(time_val, (int, float)):
                # JULD is typically days since reference date (often 1950-01-01)
                if 0 < time_val < 1000000:  # Reasonable range for JULD
                    # Convert from Julian days to datetime (assuming 1950-01-01 reference)
                    from datetime import datetime, timedelta
                    reference_date = datetime(1950, 1, 1)
                    actual_date = reference_date + timedelta(days=float(time_val))
                    return actual_date.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    return str(time_val)
            else:
                return str(time_val)
        except Exception as e:
            logger.debug(f"Time formatting error: {e}")
            return str(time_val) if time_val is not None else None
    
    def _validate_profile(self, profile: Dict) -> bool:
        """Quick validation for extracted profile"""
        try:
            # Must have valid coordinates
            if not isinstance(profile.get('lat'), (int, float)) or not isinstance(profile.get('lon'), (int, float)):
                return False
            
            # Coordinates must be in valid range
            if not (-90 <= profile['lat'] <= 90 and -180 <= profile['lon'] <= 180):
                return False
            
            # Must have some level data
            level_data = profile.get('level_data', {})
            if not level_data:
                return False
            
            # At least one variable must have non-null data
            for var_name, values in level_data.items():
                if values and any(v is not None for v in values):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _fallback_parse(self, ds: xr.Dataset, file_path: Path) -> List[Dict]:
        """Fallback to bulletproof parser for unusual files"""
        try:
            # Import the bulletproof parser
            from bulletproof_argo_ingest import BulletproofArgoParser
            
            parser = BulletproofArgoParser()
            profiles = parser.parse_file_bulletproof(file_path)
            
            logger.info(f"Fallback parsing successful for {file_path.name}: {len(profiles)} profiles")
            return profiles
            
        except Exception as e:
            logger.error(f"Fallback parsing also failed for {file_path.name}: {e}")
            return []

# Integration function for your schema
def parse_your_argo_files(file_list: List[str], max_files: Optional[int] = None) -> List[Dict]:
    """
    Parse your ARGO files using optimized parser.
    Returns all extracted profiles.
    """
    parser = OptimizedArgoParser()
    all_profiles = []
    
    files_to_process = file_list[:max_files] if max_files else file_list
    
    print(f"🚀 Processing {len(files_to_process)} ARGO files with your schema...")
    
    from tqdm import tqdm
    for file_path in tqdm(files_to_process, desc="Processing files"):
        try:
            profiles = parser.parse_your_netcdf(file_path)
            all_profiles.extend(profiles)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue
    
    print(f"✅ Extraction complete!")
    print(f"   Files processed: {parser.files_processed}")
    print(f"   Profiles extracted: {parser.profiles_extracted}")
    print(f"   Average profiles per file: {parser.profiles_extracted/max(parser.files_processed, 1):.1f}")
    
    return all_profiles

# Quick test function for your schema
def test_your_schema(sample_file: str):
    """Test the optimized parser with one of your files"""
    print(f"🧪 Testing optimized parser with your file: {sample_file}")
    
    parser = OptimizedArgoParser()
    profiles = parser.parse_your_netcdf(sample_file)
    
    if profiles:
        profile = profiles[0]
        print(f"✅ Successfully parsed {sample_file}")
        print(f"   Profile ID: {profile['profile_id']}")
        print(f"   Location: {profile['lat']:.2f}°N, {profile['lon']:.2f}°E") 
        print(f"   Time: {profile['time']}")
        print(f"   Variables: {profile['variables']}")
        print(f"   Levels: {profile['n_levels']}")
        
        # Show data sample
        for var_name, values in profile['level_data'].items():
            non_null = [v for v in values if v is not None]
            if non_null:
                print(f"   {var_name}: {len(non_null)} values, range: {min(non_null):.2f} to {max(non_null):.2f}")
        
        return True
    else:
        print(f"❌ Failed to parse {sample_file}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python optimized_argo_parser.py <sample_netcdf_file>")
        print("This will test the parser with your specific schema")
        sys.exit(1)
    
    sample_file = sys.argv[1]
    test_your_schema(sample_file)