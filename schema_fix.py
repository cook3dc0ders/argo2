# schema_fix.py
"""
Quick fix for database schema issues.
Adds missing columns to existing tables.
"""
from sqlalchemy import create_engine, text, inspect
from config import PG

def fix_database_schema():
    """Fix the database schema by adding missing columns"""
    try:
        url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
        engine = create_engine(url, pool_pre_ping=True)
        
        print("Checking and fixing database schema...")
        
        with engine.begin() as conn:
            # Get existing columns
            inspector = inspect(engine)
            
            if 'floats' in inspector.get_table_names():
                existing_columns = [col['name'] for col in inspector.get_columns('floats')]
                print(f"Existing columns: {existing_columns}")
                
                # Add missing columns
                missing_columns = []
                
                if 'data_quality_score' not in existing_columns:
                    conn.execute(text("ALTER TABLE floats ADD COLUMN data_quality_score FLOAT DEFAULT 1.0"))
                    missing_columns.append('data_quality_score')
                
                if 'extraction_method' not in existing_columns:
                    conn.execute(text("ALTER TABLE floats ADD COLUMN extraction_method TEXT DEFAULT 'unknown'"))
                    missing_columns.append('extraction_method')
                
                if 'schema_info' not in existing_columns:
                    conn.execute(text("ALTER TABLE floats ADD COLUMN schema_info JSONB DEFAULT '{}'"))
                    missing_columns.append('schema_info')
                
                if 'raw_metadata' not in existing_columns:
                    conn.execute(text("ALTER TABLE floats ADD COLUMN raw_metadata JSONB DEFAULT '{}'"))
                    missing_columns.append('raw_metadata')
                
                if 'processing_errors' not in existing_columns:
                    conn.execute(text("ALTER TABLE floats ADD COLUMN processing_errors JSONB DEFAULT '[]'"))
                    missing_columns.append('processing_errors')
                
                if 'file_size' not in existing_columns:
                    conn.execute(text("ALTER TABLE floats ADD COLUMN file_size BIGINT"))
                    missing_columns.append('file_size')
                
                if missing_columns:
                    print(f"Added missing columns: {missing_columns}")
                else:
                    print("All required columns already exist")
                
                # Create missing indexes
                try:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_data_quality ON floats(data_quality_score);
                        CREATE INDEX IF NOT EXISTS idx_extraction_method ON floats(extraction_method);
                    """))
                    print("Added missing indexes")
                except Exception as e:
                    print(f"Warning: Could not create indexes: {e}")
                
                # Create error table if it doesn't exist
                if 'ingestion_errors' not in inspector.get_table_names():
                    conn.execute(text("""
                        CREATE TABLE ingestion_errors (
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
                    print("Created ingestion_errors table")
                
            else:
                print("floats table doesn't exist - will be created by ingestion process")
                
        print("Schema fix completed successfully!")
        return True
        
    except Exception as e:
        print(f"Schema fix failed: {e}")
        return False

if __name__ == "__main__":
    fix_database_schema()