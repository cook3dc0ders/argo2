# setup.py
"""
Automated setup script for ARGO RAG PoC on Windows.
This script will set up the database, create sample data, and verify the installation.
"""
import subprocess
import sys
import time
import requests
from pathlib import Path
import json

def run_command(cmd, shell=True, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        raise

def check_docker():
    """Check if Docker is running"""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is available")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Docker is not available or not running")
    print("Please install Docker Desktop and make sure it's running")
    return False

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'pandas', 'xarray', 
        'sqlalchemy', 'psycopg2', 'sentence_transformers', 'chromadb'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required Python packages are installed")
    return True

def setup_database():
    """Set up PostgreSQL database using Docker"""
    print("\n🗄️ Setting up PostgreSQL database...")
    
    # Start PostgreSQL
    try:
        run_command("docker-compose up -d postgres")
        print("✅ PostgreSQL container started")
    except Exception as e:
        print(f"❌ Failed to start PostgreSQL: {e}")
        return False
    
    # Wait for PostgreSQL to be ready
    print("⏳ Waiting for PostgreSQL to be ready...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            result = subprocess.run([
                "docker", "exec", "argo-rag-poc-postgres-1", 
                "pg_isready", "-U", "argo_user", "-d", "argo_db"
            ], capture_output=True)
            
            if result.returncode == 0:
                print("✅ PostgreSQL is ready")
                break
        except:
            pass
        
        time.sleep(1)
        if i == 29:
            print("❌ PostgreSQL failed to start within 30 seconds")
            return False
    
    # Create schema
    print("🔧 Creating database schema...")
    try:
        # Read schema file
        schema_path = Path("schema.sql")
        if not schema_path.exists():
            print("❌ schema.sql file not found")
            return False
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema
        cmd = f'docker exec -i argo-rag-poc-postgres-1 psql -U argo_user -d argo_db'
        result = subprocess.run(cmd, input=schema_sql, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Database schema created successfully")
            return True
        else:
            print(f"❌ Failed to create schema: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating schema: {e}")
        return False

def create_sample_data():
    """Create sample ARGO data for testing"""
    print("\n📊 Creating sample data...")
    
    try:
        # Import and run the sample data creator
        from create_sample_data import main as create_data_main
        sample_dir = create_data_main()
        print(f"✅ Sample data created in {sample_dir}")
        return sample_dir
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return None

def ingest_sample_data(sample_dir):
    """Run ingestion on sample data"""
    print(f"\n📥 Ingesting sample data from {sample_dir}...")
    
    try:
        cmd = f"python ingest.py --dir {sample_dir} --parquet-dir ./parquet_store"
        run_command(cmd)
        print("✅ Sample data ingested successfully")
        return True
    except Exception as e:
        print(f"❌ Error ingesting sample data: {e}")
        return False

def test_api():
    """Test the API endpoints"""
    print("\n🧪 Testing API endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API health check passed")
            print(f"   Database: {health_data.get('database')}")
            print(f"   Embeddings: {health_data.get('embeddings')}")
            print(f"   OpenAI: {health_data.get('openai')}")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not accessible: {e}")
        return False
    
    # Test query endpoint
    try:
        test_query = {"query": "temperature profiles", "k": 3}
        response = requests.post("http://localhost:8000/query", json=test_query, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query test passed - found {len(data.get('results', []))} results")
            return True
        else:
            print(f"❌ Query test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Query test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 ARGO RAG PoC Setup Script")
    print("=" * 50)
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    
    if not check_docker():
        return False
    
    if not check_python_packages():
        print("\n💡 To install missing packages, run:")
        print("   pip install -r requirements.txt")
        return False
    
    # Setup database
    if not setup_database():
        return False
    
    # Create and ingest sample data
    sample_dir = create_sample_data()
    if not sample_dir:
        return False
    
    if not ingest_sample_data(sample_dir):
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start the application:")
    print("   1. Start RAG server: uvicorn rag_server:app --reload --port 8000")
    print("   2. Start dashboard: streamlit run dashboard.py")
    print("\n📱 Then open your browser to:")
    print("   - Dashboard: http://localhost:8501")
    print("   - API docs: http://localhost:8000/docs")
    
    return True

def verify_installation():
    """Verify that everything is working"""
    print("\n🔍 Verifying installation...")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ RAG server is running")
            
            # Run API tests
            if test_api():
                print("\n🎉 Installation verification completed successfully!")
                print("Your ARGO RAG PoC is ready to use!")
                return True
        else:
            print("❌ RAG server responded with error")
    except requests.exceptions.ConnectionError:
        print("❌ RAG server is not running")
        print("Start it with: uvicorn rag_server:app --reload --port 8000")
    except Exception as e:
        print(f"❌ Error checking server: {e}")
    
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup ARGO RAG PoC")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing installation")
    parser.add_argument("--skip-data", action="store_true", help="Skip sample data creation")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_installation()
    else:
        if main():
            print("\n🔍 Would you like to verify the installation? (y/n): ", end="")
            try:
                choice = input().lower().strip()
                if choice in ['y', 'yes']:
                    verify_installation()
            except KeyboardInterrupt:
                print("\nSetup completed. Run verification manually if needed.")
        else:
            print("❌ Setup failed. Please check the errors above and try again.")
            sys.exit(1)