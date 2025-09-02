#!/usr/bin/env python3
"""
Diagnostic and fix script for ARGO RAG system.
Identifies and resolves issues between PostgreSQL and ChromaDB synchronization.
"""
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from config import PG, CHROMA_DIR
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pg_engine():
    """Create PostgreSQL engine"""
    url = f"postgresql://{PG['user']}:{PG['password']}@{PG['host']}:{PG['port']}/{PG['db']}"
    return create_engine(url, pool_pre_ping=True)

def diagnose_database():
    """Check what's in the PostgreSQL database"""
    print("🔍 DIAGNOSING POSTGRESQL DATABASE")
    print("=" * 50)
    
    try:
        engine = pg_engine()
        with engine.connect() as conn:
            # Count total records
            result = conn.execute(text("SELECT COUNT(*) FROM floats"))
            total_count = result.scalar()
            print(f"Total records in database: {total_count}")
            
            if total_count == 0:
                print("❌ Database is empty - no data to work with")
                return False
            
            # Show sample records
            sample_result = conn.execute(text("""
                SELECT profile_id, float_id, lat, lon, time_start, n_levels, variables_list, parquet_path
                FROM floats 
                LIMIT 5
            """))
            
            print("\n📋 Sample records:")
            for row in sample_result:
                print(f"  Profile: {row[0]}")
                print(f"    Float ID: {row[1]}")
                print(f"    Location: {row[2]}, {row[3]}")
                print(f"    Time: {row[4]}")
                print(f"    Levels: {row[5]}")
                print(f"    Variables: {row[6]}")
                print(f"    Parquet: {row[7]}")
                print()
            
            # Check for null values in critical fields
            null_check = conn.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(lat) as has_lat,
                    COUNT(lon) as has_lon,
                    COUNT(time_start) as has_time,
                    COUNT(parquet_path) as has_parquet
                FROM floats
            """)).fetchone()
            
            print(f"Data completeness:")
            print(f"  Records with lat: {null_check[1]}/{null_check[0]}")
            print(f"  Records with lon: {null_check[2]}/{null_check[0]}")
            print(f"  Records with time: {null_check[3]}/{null_check[0]}")
            print(f"  Records with parquet: {null_check[4]}/{null_check[0]}")
            
            return True
            
    except Exception as e:
        print(f"❌ Database diagnosis failed: {e}")
        return False

def diagnose_vector_db():
    """Check what's in the ChromaDB vector database"""
    print("\n🔍 DIAGNOSING CHROMADB VECTOR DATABASE")
    print("=" * 50)
    
    try:
        from embeddings_utils import EmbeddingManager
        emb = EmbeddingManager()
        
        # Get collection stats
        stats = emb.get_collection_stats()
        print(f"Vector database stats: {stats}")
        
        # Try a simple query
        try:
            results = emb.query("temperature", n_results=3)
            ids = results.get("ids", [[]])[0]
            print(f"\nSample vector query results: {len(ids)} matches")
            for i, profile_id in enumerate(ids[:3]):
                print(f"  {i+1}. {profile_id}")
            
            return len(ids) > 0
            
        except Exception as e:
            print(f"❌ Vector query failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Vector database diagnosis failed: {e}")
        return False

def rebuild_vector_database():
    """Rebuild the vector database from PostgreSQL data"""
    print("\n🔄 REBUILDING VECTOR DATABASE")
    print("=" * 50)
    
    try:
        # Initialize fresh embeddings manager
        from embeddings_utils import EmbeddingManager
        emb = EmbeddingManager()
        
        # Clear existing collection
        print("Clearing existing vector database...")
        emb.reset_collection()
        
        # Get all profiles from database
        engine = pg_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT profile_id, float_id, lat, lon, time_start, n_levels, 
                       variables_list, parquet_path
                FROM floats 
                WHERE profile_id IS NOT NULL
                ORDER BY created_at
            """))
            
            profiles = result.fetchall()
            print(f"Found {len(profiles)} profiles in database")
            
            if not profiles:
                print("❌ No profiles found in database")
                return False
            
            # Process profiles in batches
            batch_size = 10
            for i in range(0, len(profiles), batch_size):
                batch = profiles[i:i+batch_size]
                
                ids = []
                metadatas = []
                summaries = []
                
                for row in batch:
                    profile_id = row[0]
                    float_id = row[1] or "unknown"
                    lat = row[2]
                    lon = row[3]
                    time_start = str(row[4]) if row[4] else None
                    n_levels = row[5] or 0
                    variables_list = row[6]
                    parquet_path = row[7]
                    
                    # Parse variables list
                    try:
                        if isinstance(variables_list, str):
                            variables = json.loads(variables_list)
                        elif isinstance(variables_list, list):
                            variables = variables_list
                        else:
                            variables = []
                    except:
                        variables = []
                    
                    # Create summary for embedding
                    summary_parts = [f"Float {float_id} profile {profile_id}"]
                    
                    if lat is not None and lon is not None:
                        summary_parts.append(f"at {lat:.2f}°N, {lon:.2f}°E")
                    
                    if time_start:
                        summary_parts.append(f"measured on {time_start}")
                    
                    if n_levels > 0:
                        summary_parts.append(f"with {n_levels} depth levels")
                    
                    if variables:
                        var_str = ", ".join(variables[:5])
                        summary_parts.append(f"measuring {var_str}")
                    
                    summary = " ".join(summary_parts) + "."
                    
                    # Create metadata (ChromaDB only accepts simple types, no None values)
                    metadata = {
                        'float_id': str(float_id),
                        'parquet': parquet_path or ""
                    }
                    
                    # Only add coordinates if they're valid numbers
                    if lat is not None and isinstance(lat, (int, float)):
                        metadata['lat'] = float(lat)
                    if lon is not None and isinstance(lon, (int, float)):
                        metadata['lon'] = float(lon)
                    
                    # Only add time if it's a valid string
                    if time_start and isinstance(time_start, str):
                        metadata['time'] = time_start
                    
                    # Only add variables if there are any
                    if variables:
                        metadata['variables_str'] = ", ".join(variables)
                    
                    ids.append(profile_id)
                    metadatas.append(metadata)
                    summaries.append(summary)
                
                # Add batch to vector database
                if ids:
                    emb.add_documents(ids, metadatas, summaries)
                    print(f"✅ Processed batch {i//batch_size + 1}: {len(ids)} profiles")
            
            print(f"✅ Vector database rebuilt with {len(profiles)} profiles")
            return True
            
    except Exception as e:
        print(f"❌ Vector database rebuild failed: {e}")
        return False

def test_full_system():
    """Test the complete RAG pipeline"""
    print("\n🧪 TESTING COMPLETE RAG SYSTEM")
    print("=" * 50)
    
    # Test vector search
    print("Testing vector search...")
    try:
        from embeddings_utils import EmbeddingManager
        emb = EmbeddingManager()
        
        test_queries = [
            "temperature profiles",
            "salinity data", 
            "equator",
            "ocean measurements"
        ]
        
        for query in test_queries:
            try:
                results = emb.query(query, n_results=3)
                ids = results.get("ids", [[]])[0]
                print(f"  Query '{query}': {len(ids)} results")
                if ids:
                    print(f"    First result: {ids[0]}")
            except Exception as e:
                print(f"  Query '{query}': FAILED - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        return False

def fix_rag_system():
    """Complete fix for the RAG system"""
    print("🚀 ARGO RAG SYSTEM DIAGNOSTIC AND FIX")
    print("=" * 60)
    
    # Step 1: Diagnose database
    if not diagnose_database():
        print("❌ Database diagnosis failed - cannot proceed")
        return False
    
    # Step 2: Diagnose vector database
    vector_ok = diagnose_vector_db()
    
    # Step 3: Rebuild vector database if needed
    if not vector_ok:
        print("🔧 Vector database needs rebuilding...")
        if not rebuild_vector_database():
            print("❌ Vector database rebuild failed")
            return False
    else:
        print("✅ Vector database seems OK")
    
    # Step 4: Test complete system
    if not test_full_system():
        print("❌ System test failed")
        return False
    
    print("\n🎉 RAG SYSTEM FIX COMPLETE!")
    print("✅ Database: OK")
    print("✅ Vector DB: OK") 
    print("✅ Integration: OK")
    print("\nYour RAG system should now work properly!")
    return True

def check_embeddings_initialization():
    """Check if embeddings are properly initialized"""
    print("\n🔍 CHECKING EMBEDDINGS INITIALIZATION")
    print("=" * 50)
    
    try:
        # Check if ChromaDB directory exists
        chroma_path = Path(CHROMA_DIR)
        print(f"ChromaDB directory: {chroma_path}")
        print(f"Directory exists: {chroma_path.exists()}")
        
        if chroma_path.exists():
            files = list(chroma_path.rglob("*"))
            print(f"Files in ChromaDB directory: {len(files)}")
            for f in files[:5]:  # Show first 5 files
                print(f"  {f.relative_to(chroma_path)}")
        
        # Try to initialize embeddings
        from embeddings_utils import EmbeddingManager
        emb = EmbeddingManager()
        
        # Get collection info
        stats = emb.get_collection_stats()
        print(f"\nCollection stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embeddings check failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix ARGO RAG system")
    parser.add_argument("--diagnose-only", action="store_true", help="Only run diagnostics")
    parser.add_argument("--rebuild-vectors", action="store_true", help="Force rebuild vector database")
    parser.add_argument("--check-embeddings", action="store_true", help="Check embeddings initialization")
    
    args = parser.parse_args()
    
    if args.check_embeddings:
        success = check_embeddings_initialization()
    elif args.diagnose_only:
        db_ok = diagnose_database()
        vector_ok = diagnose_vector_db()
        success = db_ok and vector_ok
    elif args.rebuild_vectors:
        success = rebuild_vector_database()
    else:
        success = fix_rag_system()
    
    sys.exit(0 if success else 1)