-- init.sql
-- This file will be automatically executed when the PostgreSQL container starts

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create the main floats table
CREATE TABLE IF NOT EXISTS floats (
    id SERIAL PRIMARY KEY,
    float_id TEXT NOT NULL,
    profile_id TEXT UNIQUE NOT NULL,
    time_start TIMESTAMP,
    time_end TIMESTAMP,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    platform_number TEXT,
    n_levels INTEGER DEFAULT 0,
    variables JSONB,
    parquet_path TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create spatial index (only if lat/lon are not null)
CREATE INDEX IF NOT EXISTS idx_float_geo 
ON floats USING GIST (ST_SetSRID(ST_Point(lon, lat), 4326))
WHERE lat IS NOT NULL AND lon IS NOT NULL;

-- Additional useful indexes
CREATE INDEX IF NOT EXISTS idx_float_id ON floats(float_id);
CREATE INDEX IF NOT EXISTS idx_profile_id ON floats(profile_id);
CREATE INDEX IF NOT EXISTS idx_time_start ON floats(time_start);
CREATE INDEX IF NOT EXISTS idx_time_range ON floats(time_start, time_end);
CREATE INDEX IF NOT EXISTS idx_coordinates ON floats(lat, lon);
CREATE INDEX IF NOT EXISTS idx_variables ON floats USING GIN(variables);
CREATE INDEX IF NOT EXISTS idx_created_at ON floats(created_at);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_floats_updated_at 
    BEFORE UPDATE ON floats 
    FOR EACH ROW 
    EXECUTE PROCEDURE update_updated_at_column();

-- Create a view for easy data exploration
CREATE OR REPLACE VIEW float_summary AS
SELECT 
    float_id,
    COUNT(*) as total_profiles,
    MIN(time_start) as first_measurement,
    MAX(time_start) as last_measurement,
    AVG(lat) as avg_lat,
    AVG(lon) as avg_lon,
    AVG(n_levels) as avg_levels,
    array_agg(DISTINCT jsonb_array_elements_text(variables)) as all_variables
FROM floats 
WHERE variables IS NOT NULL
GROUP BY float_id
ORDER BY first_measurement DESC;

-- Insert some metadata about the database setup
INSERT INTO floats (float_id, profile_id, platform_number, created_at) 
VALUES ('SYSTEM', 'INIT_MARKER', 'Database initialized', NOW())
ON CONFLICT (profile_id) DO NOTHING;

-- Print success message (will show in container logs)
DO $$
BEGIN
    RAISE NOTICE 'ARGO RAG database schema created successfully!';
    RAISE NOTICE 'Tables: %', (SELECT string_agg(tablename, ', ') FROM pg_tables WHERE schemaname = 'public');
END $$;