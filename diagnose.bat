@echo off
setlocal EnableDelayedExpansion

REM diagnose.bat
REM Diagnostic script for ARGO RAG PoC troubleshooting

echo.
echo ========================================
echo  ARGO RAG PoC - Diagnostic Script
echo ========================================
echo.

cd /d "%~dp0"

echo SYSTEM INFORMATION:
echo Current directory: %CD%
echo Current time: %DATE% %TIME%
echo.

REM Check file structure
echo [1/8] Checking project structure...
set missing_files=0
for %%f in (config.py rag_server.py dashboard.py ingest.py requirements.txt docker-compose.yml) do (
    if exist "%%f" (
        echo SUCCESS: %%f found
    ) else (
        echo ERROR: %%f missing
        set /a missing_files+=1
    )
)

if !missing_files! gtr 0 (
    echo WARNING: !missing_files! critical files are missing
)
echo.

REM Check Docker
echo [2/8] Checking Docker...
docker --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Docker not found in PATH
    echo Install Docker Desktop from: https://www.docker.com/products/docker-desktop/
) else (
    for /f "tokens=*" %%i in ('docker --version 2^>nul') do echo SUCCESS: %%i
    
    docker ps >nul 2>&1
    if !errorlevel! neq 0 (
        echo ERROR: Docker daemon not running - start Docker Desktop
    ) else (
        echo SUCCESS: Docker daemon is running
    )
)
echo.

REM Check Python
echo [3/8] Checking Python...
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python not found in PATH
    echo Install Python 3.10+ from: https://www.python.org/downloads/
) else (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo SUCCESS: %%i
    
    REM Check pip
    python -m pip --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo ERROR: pip not available
    ) else (
        echo SUCCESS: pip is available
    )
)
echo.

REM Check virtual environment
echo [4/8] Checking virtual environment...
if not exist "venv" (
    echo ERROR: Virtual environment not found
    echo Create it with: python -m venv venv
) else (
    echo SUCCESS: Virtual environment directory exists
    
    if exist "venv\Scripts\activate.bat" (
        echo SUCCESS: Activation script found
    ) else (
        echo ERROR: Activation script missing - recreate venv
    )
)
echo.

REM Check packages (if venv exists)
echo [5/8] Checking Python packages...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    
    set packages_ok=1
    for %%p in (fastapi uvicorn streamlit pandas xarray sqlalchemy psycopg2 sentence_transformers chromadb) do (
        python -c "import %%p" >nul 2>&1
        if !errorlevel! neq 0 (
            echo ERROR: Package %%p not installed
            set packages_ok=0
        ) else (
            echo SUCCESS: Package %%p installed
        )
    )
    
    if !packages_ok! equ 0 (
        echo.
        echo To install missing packages: pip install -r requirements.txt
    )
) else (
    echo SKIPPED: Virtual environment not available
)
echo.

REM Check Docker containers
echo [6/8] Checking Docker containers...
docker-compose ps
echo.

REM Check if containers are running
docker ps --filter "name=argo-postgres" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr "argo-postgres" >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: PostgreSQL container is running
    
    REM Test database connection
    docker exec argo-postgres pg_isready -U argo_user -d argo_db >nul 2>&1
    if !errorlevel! equ 0 (
        echo SUCCESS: PostgreSQL is accepting connections
    ) else (
        echo ERROR: PostgreSQL not ready for connections
    )
) else (
    echo ERROR: PostgreSQL container not running
    echo Start it with: docker-compose up -d postgres
)
echo.

REM Check ports
echo [7/8] Checking port availability...
netstat -an | findstr ":5432 " >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: Port 5432 (PostgreSQL) is in use
) else (
    echo WARNING: Port 5432 not in use - PostgreSQL may not be running
)

netstat -an | findstr ":8000 " >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: Port 8000 (RAG server) is in use
) else (
    echo INFO: Port 8000 available (RAG server not running)
)

netstat -an | findstr ":8501 " >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: Port 8501 (Dashboard) is in use
) else (
    echo INFO: Port 8501 available (Dashboard not running)
)
echo.

REM Test API if running
echo [8/8] Testing API endpoints...
python -c "import requests; print('Testing API...'); r=requests.get('http://localhost:8000/health', timeout=5); print(f'Health: {r.status_code}'); exit(0)" >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: API health endpoint responding
    
    REM Test query endpoint
    python -c "import requests, json; r=requests.post('http://localhost:8000/query', json={'query': 'test', 'k': 1}, timeout=10); print(f'Query: {r.status_code}'); exit(0)" >nul 2>&1
    if !errorlevel! equ 0 (
        echo SUCCESS: API query endpoint responding
    ) else (
        echo WARNING: Query endpoint not responding (may need data ingestion)
    )
) else (
    echo INFO: API not responding (not running or still starting)
)
echo.

REM Final recommendations
echo ========================================
echo  DIAGNOSTIC SUMMARY
echo ========================================
echo.

echo If you see errors above, try these solutions:
echo.
echo FOR DOCKER ISSUES:
echo   - Restart Docker Desktop
echo   - Run: docker-compose down --volumes
echo   - Run: docker-compose up -d postgres
echo.
echo FOR PYTHON ISSUES:
echo   - Recreate virtual environment: rmdir /s venv && python -m venv venv
echo   - Reinstall packages: venv\Scripts\activate.bat && pip install -r requirements.txt
echo.
echo FOR DATABASE ISSUES:
echo   - Reset containers: reset_docker.bat
echo   - Check logs: docker-compose logs postgres
echo.
echo FOR API ISSUES:
echo   - Check server window for error messages
echo   - Verify packages: venv\Scripts\activate.bat && python -c "import fastapi, uvicorn"
echo   - Test manually: venv\Scripts\activate.bat && python rag_server.py
echo.
echo FOR COMPLETE RESET:
echo   1. Run: reset_docker.bat
echo   2. Run: QUICKSTART_WINDOWS.bat
echo.
pause