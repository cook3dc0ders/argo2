-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create the main table
CREATE TABLE IF NOT EXISTS floats (
    id SERIAL PRIMARY KEY,
    float_id TEXT NOT NULL,
    profile_id TEXT UNIQUE,  -- Added UNIQUE constraint
    time_start TIMESTAMP,
    time_end TIMESTAMP,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    platform_number TEXT,
    n_levels INTEGER,
    variables JSONB,
    parquet_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create spatial index (only if lat/lon are not null)
CREATE INDEX IF NOT EXISTS idx_float_geo 
ON floats USING GIST (ST_SetSRID(ST_Point(lon, lat), 4326))
WHERE lat IS NOT NULL AND lon IS NOT NULL;

-- Additional useful indexes
CREATE INDEX IF NOT EXISTS idx_float_id ON floats(float_id);
CREATE INDEX IF NOT EXISTS idx_time_start ON floats(time_start);
CREATE INDEX IF NOT EXISTS idx_variables ON floats USING GIN(variables);