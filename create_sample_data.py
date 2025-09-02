# create_sample_data.py
"""
Create sample ARGO-like NetCDF files for testing the RAG system.
This generates realistic oceanographic profiles for development and testing.
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import random

def create_realistic_profile(lat, lon, date, depth_max=2000):
    """Create a realistic temperature/salinity profile"""
    # Depth levels (irregular spacing, denser near surface)
    depths = np.concatenate([
        np.linspace(0, 100, 20),      # Surface layer
        np.linspace(100, 500, 15),    # Thermocline
        np.linspace(500, depth_max, 15)  # Deep water
    ])
    
    # Temperature profile (realistic thermocline)
    surface_temp = 25 + 5 * np.sin(np.radians(lat)) + random.gauss(0, 2)
    deep_temp = 2 + random.gauss(0, 0.5)
    
    # Create thermocline structure
    temp = []
    for d in depths:
        if d < 50:
            # Mixed layer
            t = surface_temp + random.gauss(0, 0.3)
        elif d < 200:
            # Thermocline
            t = surface_temp - (surface_temp - deep_temp) * (d - 50) / 150 + random.gauss(0, 0.5)
        else:
            # Deep water
            t = deep_temp + random.gauss(0, 0.2)
        temp.append(max(t, 0))  # Temperature can't be negative
    
    # Salinity profile
    surface_sal = 35 + random.gauss(0, 0.5)
    deep_sal = 34.7 + random.gauss(0, 0.1)
    
    salinity = []
    for d in depths:
        if d < 100:
            s = surface_sal + random.gauss(0, 0.1)
        else:
            s = deep_sal + random.gauss(0, 0.05)
        salinity.append(max(s, 30))  # Reasonable salinity bounds
    
    # Add some oxygen data for variety
    oxygen = []
    for d in depths:
        if d < 100:
            o = 6 + random.gauss(0, 0.5)  # High surface oxygen
        elif d < 500:
            o = 3 + random.gauss(0, 0.3)  # Oxygen minimum zone
        else:
            o = 4 + random.gauss(0, 0.2)  # Deep water oxygen
        oxygen.append(max(o, 0))
    
    return depths, temp, salinity, oxygen

def create_sample_netcdf_file(output_path, n_profiles=3):
    """Create a multi-profile ARGO-like NetCDF file"""
    
    # Random locations and times
    locations = [
        (25.5, -80.0),    # Atlantic
        (35.2, -75.5),    # North Atlantic
        (0.0, -25.0),     # Equatorial Atlantic
    ]
    
    base_date = datetime(2023, 6, 15)
    dates = [base_date + timedelta(days=i*10) for i in range(n_profiles)]
    
    # Create data for each profile
    all_temp = []
    all_sal = []
    all_oxygen = []
    all_pres = []
    all_lats = []
    all_lons = []
    all_times = []
    
    max_levels = 0
    
    for i in range(n_profiles):
        lat, lon = locations[i % len(locations)]
        date = dates[i]
        
        depths, temp, sal, oxy = create_realistic_profile(lat, lon, date)
        max_levels = max(max_levels, len(depths))
        
        all_temp.append(temp)
        all_sal.append(sal)
        all_oxygen.append(oxy)
        all_pres.append(depths)
        all_lats.append(lat)
        all_lons.append(lon)
        all_times.append(date)
    
    # Pad arrays to same length
    for i in range(n_profiles):
        while len(all_temp[i]) < max_levels:
            all_temp[i].append(np.nan)
            all_sal[i].append(np.nan)
            all_oxygen[i].append(np.nan)
            all_pres[i].append(np.nan)
    
    # Create xarray dataset
    ds = xr.Dataset({
        'TEMP': (['N_PROF', 'N_LEVELS'], all_temp),
        'PSAL': (['N_PROF', 'N_LEVELS'], all_sal),
        'DOXY': (['N_PROF', 'N_LEVELS'], all_oxygen),
        'PRES': (['N_PROF', 'N_LEVELS'], all_pres),
        'LATITUDE': (['N_PROF'], all_lats),
        'LONGITUDE': (['N_PROF'], all_lons),
        'JULD': (['N_PROF'], [pd.Timestamp(d) for d in all_times]),
        'PLATFORM_NUMBER': (['N_PROF'], [f"590{i:04d}" for i in range(n_profiles)])
    })
    
    # Add attributes (metadata)
    ds.attrs.update({
        'title': 'Sample ARGO profiles for testing',
        'institution': 'Test Institution',
        'source': 'Simulated data',
        'date_created': datetime.now().isoformat()
    })
    
    # Add variable attributes
    ds['TEMP'].attrs.update({
        'long_name': 'Sea water temperature',
        'units': 'degree_Celsius',
        'standard_name': 'sea_water_temperature'
    })
    
    ds['PSAL'].attrs.update({
        'long_name': 'Sea water practical salinity',
        'units': 'PSU',
        'standard_name': 'sea_water_practical_salinity'
    })
    
    ds['PRES'].attrs.update({
        'long_name': 'Sea water pressure',
        'units': 'decibar',
        'standard_name': 'sea_water_pressure'
    })
    
    ds['DOXY'].attrs.update({
        'long_name': 'Dissolved oxygen',
        'units': 'micromole/kg',
        'standard_name': 'moles_of_oxygen_per_unit_mass_in_sea_water'
    })
    
    # Save file
    ds.to_netcdf(output_path)
    ds.close()
    
    return output_path

def create_single_profile_files(output_dir, n_files=5):
    """Create multiple single-profile NetCDF files"""
    output_dir = Path(output_dir)
    
    # Different ocean regions
    regions = [
        {"name": "tropical_atlantic", "lat_range": (-10, 10), "lon_range": (-50, 10)},
        {"name": "north_atlantic", "lat_range": (40, 60), "lon_range": (-60, -10)},
        {"name": "south_pacific", "lat_range": (-40, -20), "lon_range": (-150, -100)},
        {"name": "indian_ocean", "lat_range": (-20, 0), "lon_range": (60, 100)},
        {"name": "arctic", "lat_range": (70, 80), "lon_range": (-20, 20)},
    ]
    
    files_created = []
    
    for i in range(n_files):
        region = regions[i % len(regions)]
        
        # Random location within region
        lat = random.uniform(*region["lat_range"])
        lon = random.uniform(*region["lon_range"])
        
        # Random date in 2023
        base_date = datetime(2023, 1, 1)
        random_days = random.randint(0, 364)
        date = base_date + timedelta(days=random_days)
        
        # Create profile
        depths, temp, sal, oxy = create_realistic_profile(lat, lon, date)
        
        # Create single-profile dataset
        ds = xr.Dataset({
            'TEMP': (['N_LEVELS'], temp),
            'PSAL': (['N_LEVELS'], sal),
            'DOXY': (['N_LEVELS'], oxy),
            'PRES': (['N_LEVELS'], depths),
            'LATITUDE': lat,
            'LONGITUDE': lon,
            'JULD': pd.Timestamp(date),
            'PLATFORM_NUMBER': f"590{i:04d}"
        })
        
        # Add attributes
        ds.attrs.update({
            'title': f'Sample ARGO profile {i+1}',
            'region': region["name"],
            'date_created': datetime.now().isoformat()
        })
        
        # Save file
        filename = f"argo_{region['name']}_{date.strftime('%Y%m%d')}_profile{i:03d}.nc"
        filepath = output_dir / filename
        ds.to_netcdf(filepath)
        ds.close()
        
        files_created.append(filepath)
        print(f"Created: {filename}")
    
    return files_created

def main():
    """Main function to create sample data"""
    print("🌊 Creating sample ARGO data for testing...")
    
    # Create output directory
    sample_dir = Path("./sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create multi-profile file
    print("\n📁 Creating multi-profile NetCDF file...")
    multi_file = create_sample_netcdf_file(
        sample_dir / "sample_argo_multiprofile.nc", 
        n_profiles=5
    )
    print(f"✅ Created: {multi_file}")
    
    # Create single-profile files
    print("\n📁 Creating individual profile NetCDF files...")
    single_files = create_single_profile_files(sample_dir, n_files=8)
    print(f"✅ Created {len(single_files)} individual profile files")
    
    # Create a summary
    print(f"\n📊 Sample data summary:")
    print(f"   Directory: {sample_dir.absolute()}")
    print(f"   Multi-profile file: 1 file with 5 profiles")
    print(f"   Single-profile files: {len(single_files)} files")
    print(f"   Total profiles: {5 + len(single_files)}")
    
    # Instructions
    print(f"\n🚀 Next steps:")
    print(f"   1. Run ingestion: python ingest.py --dir {sample_dir}")
    print(f"   2. Start RAG server: uvicorn rag_server:app --reload --port 8000")
    print(f"   3. Start dashboard: streamlit run dashboard.py")
    
    return sample_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample ARGO data for testing")
    parser.add_argument("--output-dir", default="./sample_data", help="Output directory for sample files")
    parser.add_argument("--n-single", type=int, default=8, help="Number of single-profile files to create")
    parser.add_argument("--n-multi", type=int, default=5, help="Number of profiles in multi-profile file")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🌊 Creating sample ARGO data in {output_dir.absolute()}...")
    
    # Create multi-profile file
    if args.n_multi > 0:
        print(f"\n📁 Creating multi-profile file with {args.n_multi} profiles...")
        multi_file = create_sample_netcdf_file(
            output_dir / "sample_argo_multiprofile.nc", 
            n_profiles=args.n_multi
        )
        print(f"✅ Created: {multi_file}")
    
    # Create single-profile files
    if args.n_single > 0:
        print(f"\n📁 Creating {args.n_single} single-profile files...")
        single_files = create_single_profile_files(output_dir, n_files=args.n_single)
        print(f"✅ Created {len(single_files)} files")
    
    print(f"\n🎉 Sample data creation complete!")
    print(f"   Output directory: {output_dir.absolute()}")
    print(f"\n🚀 Next steps:")
    print(f"   1. Run: python ingest.py --dir {output_dir}")
    print(f"   2. Start server: uvicorn rag_server:app --reload --port 8000")
    print(f"   3. Start dashboard: streamlit run dashboard.py")