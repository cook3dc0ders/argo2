@echo off
setlocal EnableDelayedExpansion

REM QUICKSTART_WINDOWS.bat
REM Fixed Windows setup script for ARGO RAG PoC

echo.
echo ========================================
echo   ARGO RAG PoC - Windows Quick Start
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if we're in the right directory
if not exist "config.py" (
    echo ERROR: Please run this script from the project root directory
    echo        The directory containing config.py
    echo.
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo INFO: Working directory: %CD%
echo.

REM Check Docker
echo [1/7] Checking Docker...
docker --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Docker is not available
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    echo Make sure Docker Desktop is running before continuing
    pause
    exit /b 1
)

REM Check if Docker daemon is running
docker ps >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Docker daemon is not running
    echo Please start Docker Desktop and wait for it to fully start
    pause
    exit /b 1
)

echo SUCCESS: Docker is available and running
echo.

REM Check Python
echo [2/7] Checking Python...
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python is not available
    echo Please install Python 3.10+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo SUCCESS: Python !PYTHON_VERSION! is available
echo.

REM Create virtual environment if it doesn't exist
echo [3/7] Setting up Python environment...
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create virtual environment
        echo Try running as administrator or check Python installation
        pause
        exit /b 1
    )
    echo SUCCESS: Virtual environment created
) else (
    echo INFO: Virtual environment already exists
)

REM Activate virtual environment and install packages
echo Installing/updating Python packages...
call venv\Scripts\activate.bat || (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
if !errorlevel! neq 0 (
    echo ERROR: Failed to install packages
    echo Trying alternative installation method...
    python -m pip install --only-binary=all -r requirements.txt
    if !errorlevel! neq 0 (
        echo ERROR: Package installation failed completely
        echo Try installing packages manually:
        echo   venv\Scripts\activate.bat
        echo   pip install fastapi uvicorn streamlit pandas xarray sqlalchemy psycopg2-binary
        pause
        exit /b 1
    )
)
echo SUCCESS: Python packages installed
echo.

REM Create .env file if it doesn't exist
echo [4/7] Configuring environment...
if not exist ".env" (
    echo Creating configuration file...
    (
        echo # ARGO RAG PoC Configuration
        echo PG_USER=argo_user
        echo PG_PASS=argo_pass
        echo PG_DB=argo_db
        echo PG_HOST=localhost
        echo PG_PORT=5432
        echo PARQUET_DIR=./parquet_store
        echo CHROMA_DIR=./chroma_db
        echo EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
        echo # Optional: Add your OpenAI API key below for enhanced responses
        echo # OPENAI_API_KEY=your_key_here
    ) > .env
    echo SUCCESS: Configuration file created
) else (
    echo INFO: Configuration file already exists
)
echo.

REM Stop any existing containers and start fresh
echo [5/7] Setting up database...

REM Clean up any conflicting Docker Compose files
if exist "docker-compose.yml" if exist "compose.yml" (
    echo WARNING: Both docker-compose.yml and compose.yml found
    echo Renaming docker-compose.yml to avoid conflicts...
    rename "docker-compose.yml" "docker-compose.yml.backup" >nul 2>&1
)

echo Stopping any existing containers...
docker-compose down --volumes --remove-orphans >nul 2>&1

REM Remove any stuck containers
docker stop argo-postgres >nul 2>&1
docker rm argo-postgres >nul 2>&1

echo Starting PostgreSQL container...
docker-compose up -d postgres
if !errorlevel! neq 0 (
    echo ERROR: Failed to start PostgreSQL container
    echo Checking Docker Compose version...
    docker-compose --version
    echo.
    echo Checking for conflicting containers...
    docker ps -a
    echo.
    echo Try running: docker-compose up postgres
    pause
    exit /b 1
)

REM Wait for PostgreSQL to be ready with better feedback
echo Waiting for PostgreSQL to initialize...
set /a count=0
:wait_loop
timeout /t 3 /nobreak >nul
set /a count+=1

REM Check if container is still running
docker ps --filter "name=argo-postgres" --format "{{.Names}}" | findstr "argo-postgres" >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: PostgreSQL container stopped unexpectedly
    echo Checking container logs...
    docker logs argo-postgres
    pause
    exit /b 1
)

REM Check if PostgreSQL is ready for connections
docker exec argo-postgres pg_isready -U argo_user -d argo_db >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: PostgreSQL is ready
    
    REM Additional stability check - wait a bit more and test actual connection
    echo Performing stability check...
    timeout /t 5 /nobreak >nul
    python -c "from config import PG; from sqlalchemy import create_engine, text; import time; time.sleep(2); engine = create_engine(f'postgresql://{PG[\"user\"]}:{PG[\"password\"]}@{PG[\"host\"]}:{PG[\"port\"]}/{PG[\"db\"]}'); conn = engine.connect(); conn.execute(text('SELECT 1')); print('Connection stable')" >nul 2>&1
    if !errorlevel! equ 0 (
        echo SUCCESS: Database connection is stable
        goto db_ready
    ) else (
        echo WARNING: Database connection unstable, waiting longer...
        timeout /t 10 /nobreak >nul
        goto db_ready
    )
)

if !count! gtr 40 (
    echo ERROR: PostgreSQL failed to start within 120 seconds
    echo Checking container status and logs...
    docker ps -a --filter "name=argo-postgres"
    echo.
    echo Container logs:
    docker logs argo-postgres
    pause
    exit /b 1
)

echo Attempt !count!/40: Still waiting for PostgreSQL...
goto wait_loop

:db_ready
echo.

REM Create sample data if it doesn't exist
echo [6/7] Preparing sample data...
if not exist "sample_data" (
    echo Creating sample ARGO data...
    python create_sample_data.py --output-dir sample_data --n-single 5 --n-multi 3
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create sample data
        pause
        exit /b 1
    )
    echo SUCCESS: Sample data created
) else (
    echo INFO: Sample data directory already exists
)

REM Always try to ingest data to ensure database is populated
echo Checking if database has data...
python -c "from config import PG; from sqlalchemy import create_engine, text; engine = create_engine(f'postgresql://{PG[\"user\"]}:{PG[\"password\"]}@{PG[\"host\"]}:{PG[\"port\"]}/{PG[\"db\"]}', connect_args={'connect_timeout': 10}); conn = engine.connect(); result = conn.execute(text('SELECT COUNT(*) FROM floats')); print(f'Records: {result.scalar()}'); exit(0 if result.scalar() > 0 else 1)" 2>nul
if !errorlevel! neq 0 (
    echo Database appears empty - preparing for ingestion...
    echo Setting up database schema...
    python simple_database_setup.py
    if !errorlevel! neq 0 (
        echo ERROR: Failed to setup database schema
        echo Trying basic database test...
        python simple_database_setup.py --test-only
        pause
        exit /b 1
    )
    
    echo Running ultra-robust ingestion...
    python ultra_robust_ingest.py --dir sample_data --parquet-dir parquet_store --verbose
    if !errorlevel! neq 0 (
        echo ERROR: Ultra-robust ingestion failed
        echo Trying fallback method...
        python ingest.py --dir sample_data --parquet-dir parquet_store
        if !errorlevel! neq 0 (
            echo ERROR: All ingestion methods failed
            echo Running database diagnostics...
            python simple_database_setup.py --test-only
            pause
            exit /b 1
        )
    )
    echo SUCCESS: Sample data ingested
) else (
    echo INFO: Database already contains data
)
echo.

REM Start services
echo [7/7] Starting application services...
echo.
echo Starting RAG server...
start "ARGO RAG Server" /D "%CD%" cmd /k "venv\Scripts\activate.bat && echo Starting RAG server on port 8000... && python -m uvicorn rag_server:app --reload --port 8000 --host 0.0.0.0"

REM Wait for RAG server to start
echo Waiting for RAG server to start...
set /a count=0
:api_wait_loop
timeout /t 2 /nobreak >nul
set /a count+=1

REM Test if API is responding
python -c "import requests; requests.get('http://localhost:8000/health', timeout=3)" >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: RAG server is responding
    goto api_ready
)

if !count! gtr 15 (
    echo WARNING: RAG server taking longer than expected to start
    echo Continuing anyway - it might still be initializing
    goto api_ready
)

echo Attempt !count!/15: Waiting for RAG server...
goto api_wait_loop

:api_ready

echo Starting Streamlit dashboard...
start "ARGO Dashboard" /D "%CD%" cmd /k "venv\Scripts\activate.bat && echo Starting dashboard on port 8501... && streamlit run dashboard.py --server.port 8501"


echo.
echo ========================================
echo   SUCCESS: ARGO RAG PoC Started!
echo ========================================
echo.
echo Your application is starting up...
echo Please wait 10-15 seconds for all services to be ready.
echo.
echo WEB INTERFACES:
echo   Dashboard:     http://localhost:8501
echo   API docs:      http://localhost:8000/docs
echo   Health check:  http://localhost:8000/health
echo   Database:      http://localhost:5050 (admin/admin)
echo.
echo EXAMPLE QUERIES to try in the dashboard:
echo   - "temperature profiles in Atlantic Ocean"
echo   - "salinity data from 2023"
echo   - "profiles with oxygen measurements"
echo   - "floats near the equator"
echo.
echo TO STOP SERVICES:
echo   1. Close the server and dashboard windows that opened
echo   2. Run: docker-compose down
echo.
echo Press any key to open the dashboard in your browser...
pause >nul

REM Open dashboard in browser
timeout /t 5 /nobreak >nul
start http://localhost:8501

echo.
echo Dashboard should open in your browser shortly.
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo This window can be closed once the services are running.
pause