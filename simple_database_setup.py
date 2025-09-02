#!/usr/bin/env python3
"""
Simple database setup that just ensures basic table exists without conflicts.
"""
import time
import sys
from sqlalchemy import create_engine, text
from config import PG

def create_engine_with_retry():
    """Create engine with connection retry"""
    url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
    
    for attempt in range(10):
        try:
            engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"Database connected successfully")
            return engine
        except Exception as e:
            if attempt < 9:
                print(f"Connection attempt {attempt + 1}/10 failed, retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"All connection attempts failed: {e}")
                return None
    return None

def setup_minimal_schema():
    """Set up the minimal required schema"""
    engine = create_engine_with_retry()
    if not engine:
        return False
    
    try:
        with engine.begin() as conn:
            # Create table with minimal required columns
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS floats (
                    id SERIAL PRIMARY KEY,
                    float_id TEXT,
                    profile_id TEXT,
                    source_file TEXT,
                    time_start TIMESTAMP,
                    lat DOUBLE PRECISION,
                    lon DOUBLE PRECISION,
                    n_levels INTEGER DEFAULT 0,
                    parquet_path TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """))
            
            # Add unique constraint if it doesn't exist
            try:
                conn.execute(text("ALTER TABLE floats ADD CONSTRAINT floats_profile_id_unique UNIQUE (profile_id)"))
            except:
                pass  # Constraint may already exist
            
            # Add optional columns
            optional_columns = [
                "ALTER TABLE floats ADD COLUMN IF NOT EXISTS time_end TIMESTAMP",
                "ALTER TABLE floats ADD COLUMN IF NOT EXISTS platform_number TEXT", 
                "ALTER TABLE floats ADD COLUMN IF NOT EXISTS variables JSONB DEFAULT '[]'::jsonb",
                "ALTER TABLE floats ADD COLUMN IF NOT EXISTS variables_list JSONB DEFAULT '[]'::jsonb",
                "ALTER TABLE floats ADD COLUMN IF NOT EXISTS raw_metadata JSONB DEFAULT '{}'::jsonb",
                "ALTER TABLE floats ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW()"
            ]
            
            for sql in optional_columns:
                try:
                    conn.execute(text(sql))
                except Exception as e:
                    print(f"Warning: Could not add column: {e}")
            
            # Create basic indexes (ignore errors)
            basic_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_floats_profile_id ON floats(profile_id)",
                "CREATE INDEX IF NOT EXISTS idx_floats_float_id ON floats(float_id)",
                "CREATE INDEX IF NOT EXISTS idx_floats_time ON floats(time_start)"
            ]
            
            for idx_sql in basic_indexes:
                try:
                    conn.execute(text(idx_sql))
                except:
                    pass
            
            print("Database schema setup completed")
            return True
            
    except Exception as e:
        print(f"Error setting up schema: {e}")
        return False

def test_insert():
    """Test if we can insert data"""
    engine = create_engine_with_retry()
    if not engine:
        return False
    
    try:
        with engine.begin() as conn:
            # Test insert
            conn.execute(text("""
                INSERT INTO floats (float_id, profile_id, lat, lon, n_levels) 
                VALUES ('test', 'test_001', 0.0, 0.0, 1)
                ON CONFLICT (profile_id) DO NOTHING
            """))
            
            # Test select
            result = conn.execute(text("SELECT COUNT(*) FROM floats"))
            count = result.scalar()
            print(f"Database test successful - {count} records found")
            
            # Clean up test record
            conn.execute(text("DELETE FROM floats WHERE profile_id = 'test_001'"))
            return True
            
    except Exception as e:
        print(f"Database test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Only test connection")
    args = parser.parse_args()
    
    if args.test_only:
        success = test_insert()
    else:
        success = setup_minimal_schema() and test_insert()
    
    sys.exit(0 if success else 1)