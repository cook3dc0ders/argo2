#!/usr/bin/env python3
"""
Database reset utility for ARGO RAG system.
Cleans up schema conflicts and prepares database for ultra-robust ingestion.
"""
import sys
from sqlalchemy import create_engine, text
from config import PG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pg_engine():
    """Create PostgreSQL engine"""
    url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
    return create_engine(url, pool_pre_ping=True)

def reset_database_schema():
    """Reset the database schema to be compatible with ultra-robust ingestion"""
    try:
        engine = pg_engine()
        
        with engine.begin() as conn:
            print("Dropping existing floats table and recreating...")
            
            # Drop table and all dependent objects
            conn.execute(text("DROP TABLE IF EXISTS floats CASCADE"))
            
            # Create fresh table with complete schema
            conn.execute(text("""
                CREATE TABLE floats (
                    id SERIAL PRIMARY KEY,
                    float_id TEXT,
                    profile_id TEXT UNIQUE NOT NULL,
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
            
            # Create basic indexes
            indexes = [
                "CREATE INDEX idx_floats_profile_id ON floats(profile_id)",
                "CREATE INDEX idx_floats_float_id ON floats(float_id)",
                "CREATE INDEX idx_floats_time ON floats(time_start)",
                "CREATE INDEX idx_floats_location ON floats(lat, lon)"
            ]
            
            for idx_sql in indexes:
                try:
                    conn.execute(text(idx_sql))
                except Exception as e:
                    logger.warning(f"Could not create index: {e}")
            
            # Try to create GIN indexes for JSONB columns
            try:
                conn.execute(text("CREATE INDEX idx_floats_variables ON floats USING GIN(variables)"))
                conn.execute(text("CREATE INDEX idx_floats_variables_list ON floats USING GIN(variables_list)"))
            except Exception as e:
                logger.warning(f"Could not create GIN indexes: {e}")
            
            print("Database schema reset successfully")
            return True
            
    except Exception as e:
        print(f"Error resetting database schema: {e}")
        return False

def clear_vector_database():
    """Clear the ChromaDB vector database"""
    try:
        from embeddings_utils import EmbeddingManager
        emb = EmbeddingManager()
        emb.reset_collection()
        print("Vector database cleared successfully")
        return True
    except Exception as e:
        print(f"Error clearing vector database: {e}")
        return False

#!/usr/bin/env python3
"""
Database reset utility for ARGO RAG system.
Handles database connection issues and schema resets robustly.
"""
import sys
import time
from sqlalchemy import create_engine, text
from config import PG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pg_engine():
    """Create PostgreSQL engine with connection retries"""
    url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
    return create_engine(url, pool_pre_ping=True, pool_recycle=300, pool_timeout=20)

def wait_for_database(max_attempts=30):
    """Wait for database to be ready with retries"""
    for attempt in range(max_attempts):
        try:
            engine = pg_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"Database ready after {attempt + 1} attempts")
            return True
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1}/{max_attempts}: Database not ready, waiting...")
                time.sleep(2)
            else:
                print(f"Database failed to connect after {max_attempts} attempts: {e}")
                return False
    return False

def reset_database_schema():
    """Reset the database schema to be compatible with ultra-robust ingestion"""
    try:
        if not wait_for_database():
            return False
        
        engine = pg_engine()
        
        with engine.begin() as conn:
            print("Dropping existing floats table and recreating...")
            
            # Drop table and all dependent objects
            conn.execute(text("DROP TABLE IF EXISTS floats CASCADE"))
            
            # Create fresh table with complete schema
            conn.execute(text("""
                CREATE TABLE floats (
                    id SERIAL PRIMARY KEY,
                    float_id TEXT,
                    profile_id TEXT UNIQUE NOT NULL,
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
            
            # Create basic indexes
            indexes = [
                "CREATE INDEX idx_floats_profile_id ON floats(profile_id)",
                "CREATE INDEX idx_floats_float_id ON floats(float_id)",
                "CREATE INDEX idx_floats_time ON floats(time_start)",
                "CREATE INDEX idx_floats_location ON floats(lat, lon)"
            ]
            
            for idx_sql in indexes:
                try:
                    conn.execute(text(idx_sql))
                except Exception as e:
                    logger.warning(f"Could not create index: {e}")
            
            # Try to create GIN indexes for JSONB columns
            try:
                conn.execute(text("CREATE INDEX idx_floats_variables ON floats USING GIN(variables)"))
                conn.execute(text("CREATE INDEX idx_floats_variables_list ON floats USING GIN(variables_list)"))
            except Exception as e:
                logger.warning(f"Could not create GIN indexes: {e}")
            
            print("Database schema reset successfully")
            return True
            
    except Exception as e:
        print(f"Error resetting database schema: {e}")
        return False

def clear_vector_database():
    """Clear the ChromaDB vector database"""
    try:
        from embeddings_utils import EmbeddingManager
        emb = EmbeddingManager()
        emb.reset_collection()
        print("Vector database cleared successfully")
        return True
    except Exception as e:
        print(f"Error clearing vector database: {e}")
        return False

def quick_ingestion_test():
    """Quick test to verify database can accept data"""
    try:
        if not wait_for_database():
            return False
        
        engine = pg_engine()
        with engine.begin() as conn:
            # Insert a test record
            conn.execute(text("""
                INSERT INTO floats (float_id, profile_id, lat, lon, n_levels) 
                VALUES ('test', 'test_profile_001', 0.0, 0.0, 1)
                ON CONFLICT (profile_id) DO NOTHING
            """))
            
            # Verify it was inserted
            result = conn.execute(text("SELECT COUNT(*) FROM floats WHERE profile_id = 'test_profile_001'"))
            count = result.scalar()
            
            if count > 0:
                print("Database ingestion test passed")
                # Clean up test record
                conn.execute(text("DELETE FROM floats WHERE profile_id = 'test_profile_001'"))
                return True
            else:
                print("Database ingestion test failed")
                return False
                
    except Exception as e:
        print(f"Database test failed: {e}")
        return False

def main():
    """Main reset function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reset ARGO database schema")
    parser.add_argument("--clear-vectors", action="store_true", help="Also clear vector database")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--test-only", action="store_true", help="Only test database connection")
    
    args = parser.parse_args()
    
    if args.test_only:
        print("Testing database connection...")
        if quick_ingestion_test():
            print("✅ Database test passed")
            return True
        else:
            print("❌ Database test failed")
            return False
    
    if not args.confirm:
        print("This will DELETE ALL DATA in the floats table!")
        response = input("Are you sure you want to continue? (type 'yes' to confirm): ")
        if response.lower() != 'yes':
            print("Operation cancelled")
            return False
    
    print("Resetting database schema...")
    
    # Reset main database
    if reset_database_schema():
        print("✅ Database schema reset successfully")
        
        # Test the reset
        if quick_ingestion_test():
            print("✅ Database test passed after reset")
        else:
            print("⚠️ Database test failed after reset")
    else:
        print("❌ Database schema reset failed")
        return False
    
    # Clear vector database if requested
    if args.clear_vectors:
        if clear_vector_database():
            print("✅ Vector database cleared successfully")
        else:
            print("⚠️ Vector database clear failed")
    
    print("\n🎉 Reset complete! You can now run ingestion again.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)