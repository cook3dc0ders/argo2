#!/usr/bin/env python3
"""
Script to populate Railway PostgreSQL database with sample ARGO data.
Run this after deploying your database to Railway.
"""
import psycopg2
import json
import os
import sys
from datetime import datetime

def get_database_url():
    """Get database URL from environment or command line"""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url and len(sys.argv) > 1:
        database_url = sys.argv[1]
    
    if not database_url:
        print("Error: DATABASE_URL not found in environment")
        print("Usage: python populate_railway_db.py [DATABASE_URL]")
        print("Or set DATABASE_URL environment variable")
        sys.exit(1)
    
    return database_url

def create_sample_profiles():
    """Create sample ARGO profile data"""
    return [
        {
            "float_id": "1901393",
            "profile_id": "argo_atlantic_001",
            "source_file": "sample_data/atlantic_profile_001.nc",
            "lat": 35.5,
            "lon": -75.2,
            "time_start": "2023-06-15 12:00:00",
            "n_levels": 50,
            "variables_list": json.dumps(["TEMP", "PSAL", "DOXY"]),
            "raw_metadata": json.dumps({
                "region": "North Atlantic",
                "max_depth": 2000,
                "temperature_range": "2.5-25.3°C",
                "salinity_range": "34.2-36.8 PSU",
                "data_quality": "good"
            })
        },
        {
            "float_id": "1901394",
            "profile_id": "argo_equatorial_001", 
            "source_file": "sample_data/equatorial_profile_001.nc",
            "lat": 0.5,
            "lon": -25.0,
            "time_start": "2023-07-01 06:00:00",
            "n_levels": 75,
            "variables_list": json.dumps(["TEMP", "PSAL"]),
            "raw_metadata": json.dumps({
                "region": "Equatorial Atlantic",
                "max_depth": 1800,
                "temperature_range": "4.1-28.7°C",
                "salinity_range": "34.9-36.1 PSU",
                "data_quality": "excellent"
            })
        },
        {
            "float_id": "1901395",
            "profile_id": "argo_arabian_001",
            "source_file": "sample_data/arabian_profile_001.nc", 
            "lat": 25.2,
            "lon": 65.8,
            "time_start": "2023-07-15 18:30:00",
            "n_levels": 60,
            "variables_list": json.dumps(["TEMP", "PSAL", "DOXY", "CHLA"]),
            "raw_metadata": json.dumps({
                "region": "Arabian Sea",
                "max_depth": 1500,
                "temperature_range": "15.2-29.1°C",
                "salinity_range": "35.1-36.9 PSU",
                "data_quality": "good",
                "special_features": "monsoon_influenced"
            })
        },
        {
            "float_id": "1901396",
            "profile_id": "argo_indian_001",
            "source_file": "sample_data/indian_profile_001.nc",
            "lat": -20.3,
            "lon": 57.5,
            "time_start": "2023-08-10 09:45:00",
            "n_levels": 80,
            "variables_list": json.dumps(["TEMP", "PSAL", "DOXY", "NITRATE"]),
            "raw_metadata": json.dumps({
                "region": "Indian Ocean",
                "max_depth": 2200,
                "temperature_range": "3.8-26.9°C",
                "salinity_range": "34.5-35.2 PSU",
                "data_quality": "excellent",
                "bgc_measurements": True
            })
        },
        {
            "float_id": "1901397",
            "profile_id": "argo_north_atlantic_gyre_001",
            "source_file": "sample_data/north_atlantic_gyre_001.nc",
            "lat": 45.7,
            "lon": -30.1, 
            "time_start": "2023-09-05 14:20:00",
            "n_levels": 65,
            "variables_list": json.dumps(["TEMP", "PSAL", "DOXY", "CHLA", "BBP"]),
            "raw_metadata": json.dumps({
                "region": "North Atlantic Gyre",
                "max_depth": 1900,
                "temperature_range": "8.2-22.1°C", 
                "salinity_range": "35.0-36.2 PSU",
                "data_quality": "excellent",
                "bgc_measurements": True,
                "optical_measurements": True
            })
        },
        {
            "float_id": "1901398",
            "profile_id": "argo_mediterranean_001",
            "source_file": "sample_data/mediterranean_001.nc",
            "lat": 40.5,
            "lon": 15.2,
            "time_start": "2023-09-20 11:15:00",
            "n_levels": 45,
            "variables_list": json.dumps(["TEMP", "PSAL", "DOXY"]),
            "raw_metadata": json.dumps({
                "region": "Mediterranean Sea",
                "max_depth": 1200,
                "temperature_range": "13.8-24.5°C",
                "salinity_range": "37.5-39.1 PSU",
                "data_quality": "good",
                "special_features": "high_salinity"
            })
        },
        {
            "float_id": "1901399", 
            "profile_id": "argo_southern_ocean_001",
            "source_file": "sample_data/southern_ocean_001.nc",
            "lat": -55.8,
            "lon": 140.2,
            "time_start": "2023-10-01 08:00:00",
            "n_levels": 90,
            "variables_list": json.dumps(["TEMP", "PSAL", "DOXY", "NITRATE", "SILICATE"]),
            "raw_metadata": json.dumps({
                "region": "Southern Ocean",
                "max_depth": 2500,
                "temperature_range": "-1.2-8.7°C",
                "salinity_range": "33.8-34.7 PSU", 
                "data_quality": "excellent",
                "bgc_measurements": True,
                "special_features": "antarctic_waters"
            })
        },
        {
            "float_id": "1901400",
            "profile_id": "argo_pacific_gyre_001", 
            "source_file": "sample_data/pacific_gyre_001.nc",
            "lat": 32.1,
            "lon": -145.8,
            "time_start": "2023-10-15 16:45:00",
            "n_levels": 70,
            "variables_list": json.dumps(["TEMP", "PSAL", "DOXY", "CHLA", "CDOM"]),
            "raw_metadata": json.dumps({
                "region": "North Pacific Gyre",
                "max_depth": 2000,
                "temperature_range": "4.5-22.8°C",
                "salinity_range": "33.9-35.8 PSU",
                "data_quality": "excellent",
                "bgc_measurements": True,
                "optical_measurements": True,
                "special_features": "oligotrophic_waters"
            })
        }
    ]

def populate_database(database_url):
    """Populate the database with sample profiles"""
    try:
        print("Connecting to Railway PostgreSQL database...")
        conn = psycopg2.connect(database_url)
        
        # Test connection
        with conn.cursor() as cursor:
            cursor.execute("SELECT version()")
            version = cursor.fetchone()
            print(f"Connected to: {version[0]}")
        
        # Check if table exists and create if needed
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'floats'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print("Creating floats table...")
                cursor.execute("""
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
                        variables_list JSONB DEFAULT '[]'::jsonb,
                        parquet_path TEXT,
                        raw_metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX idx_floats_profile_id ON floats(profile_id);")
                cursor.execute("CREATE INDEX idx_floats_time ON floats(time_start);")
                cursor.execute("CREATE INDEX idx_floats_location ON floats(lat, lon);")
                cursor.execute("CREATE INDEX idx_floats_variables ON floats USING GIN(variables_list);")
                
                conn.commit()
                print("Table created successfully!")
            else:
                print("Table already exists, proceeding with data insertion...")
        
        # Insert sample profiles
        profiles = create_sample_profiles()
        print(f"Inserting {len(profiles)} sample profiles...")
        
        with conn.cursor() as cursor:
            for i, profile in enumerate(profiles, 1):
                try:
                    cursor.execute("""
                        INSERT INTO floats (
                            float_id, profile_id, source_file, lat, lon, 
                            time_start, n_levels, variables_list, raw_metadata
                        ) VALUES (
                            %(float_id)s, %(profile_id)s, %(source_file)s, 
                            %(lat)s, %(lon)s, %(time_start)s, %(n_levels)s, 
                            %(variables_list)s, %(raw_metadata)s
                        )
                        ON CONFLICT (profile_id) DO UPDATE SET
                            updated_at = NOW(),
                            raw_metadata = EXCLUDED.raw_metadata
                    """, profile)
                    
                    print(f"  {i}/{len(profiles)}: Inserted {profile['profile_id']}")
                    
                except Exception as e:
                    print(f"  Error inserting {profile['profile_id']}: {e}")
                    continue
        
        conn.commit()
        
        # Verify insertion
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM floats")
            count = cursor.fetchone()[0]
            print(f"\nDatabase now contains {count} profiles")
            
            # Show sample of inserted data
            cursor.execute("""
                SELECT profile_id, lat, lon, jsonb_array_length(variables_list) as var_count
                FROM floats 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            
            print("\nRecent profiles:")
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]:.1f}°N, {row[2]:.1f}°E, {row[3]} variables")
        
        conn.close()
        print("\n
