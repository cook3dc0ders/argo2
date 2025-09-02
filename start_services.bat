@echo off
setlocal EnableDelayedExpansion

REM start_services.bat
REM Fixed Windows batch script to start ARGO RAG PoC services

echo.
echo ========================================
echo  ARGO RAG PoC - Service Startup
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Verify we're in the right place
if not exist "config.py" (
    echo ERROR: config.py not found in current directory
    echo Please run this script from the project root directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Check Docker
echo [1/4] Checking Docker availability...
docker --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Docker is not available
    echo Please install Docker Desktop and make sure it's running
    pause
    exit /b 1
)

docker ps >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Docker daemon is not running
    echo Please start Docker Desktop and wait for it to be ready
    pause
    exit /b 1
)

echo SUCCESS: Docker is available and running
echo.

REM Check virtual environment
echo [2/4] Checking Python environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run QUICKSTART_WINDOWS.bat first to set up the environment
    pause
    exit /b 1
)

echo SUCCESS: Virtual environment found
echo.

REM Stop any existing containers and start database
echo [3/4] Managing database services...
echo Stopping any existing containers...
docker-compose down >nul 2>&1

echo Starting PostgreSQL database...
docker-compose up -d postgres
if !errorlevel! neq 0 (
    echo ERROR: Failed to start PostgreSQL
    echo Checking Docker Compose status...
    docker-compose ps
    echo.
    echo Try running: docker-compose logs postgres
    pause
    exit /b 1
)

REM Wait for database to be ready
echo Waiting for database to be ready...
set /a db_count=0
:db_wait_loop
timeout /t 3 /nobreak >nul
set /a db_count+=1

docker exec argo-postgres pg_isready -U argo_user -d argo_db >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: PostgreSQL is ready
    goto db_is_ready
)

if !db_count! gtr 20 (
    echo ERROR: Database failed to start within 60 seconds
    echo Checking container status...
    docker-compose ps
    echo.
    echo Checking logs...
    docker-compose logs postgres
    pause
    exit /b 1
)

echo Attempt !db_count!/20: Database still starting...
goto db_wait_loop

:db_is_ready
echo.

REM Start application services
echo [4/4] Starting application services...

REM Kill any existing processes on our ports
echo Checking for existing processes on ports 8000 and 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 "') do (
    echo Stopping process on port 8000: %%a
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501 "') do (
    echo Stopping process on port 8501: %%a
    taskkill /F /PID %%a >nul 2>&1
)

timeout /t 2 /nobreak >nul

REM Start RAG server
echo Starting RAG server...
start "ARGO RAG Server - PORT 8000" cmd /k ^
    "title ARGO RAG Server && echo Starting RAG server... && cd /d "%CD%" && venv\Scripts\activate.bat && python -m uvicorn rag_server:app --reload --port 8000 --host 0.0.0.0"

REM Wait for RAG server
echo Waiting for RAG server to initialize...
set /a api_count=0
:api_wait_loop
timeout /t 3 /nobreak >nul
set /a api_count+=1

REM Test API health
python -c "import requests; r=requests.get('http://localhost:8000/health', timeout=5); exit(0 if r.status_code==200 else 1)" >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: RAG server is responding
    goto api_is_ready
)

if !api_count! gtr 15 (
    echo WARNING: RAG server taking longer than expected
    echo Continuing with dashboard startup...
    goto api_is_ready
)

echo Attempt !api_count!/15: Waiting for RAG server...
goto api_wait_loop

:api_is_ready

REM Start Streamlit dashboard
echo Starting Streamlit dashboard...
start "ARGO Dashboard - PORT 8501" cmd /k ^
    "title ARGO Dashboard && echo Starting dashboard... && cd /d "%CD%" && venv\Scripts\activate.bat && streamlit run dashboard.py --server.port 8501"

echo.
echo ========================================
echo   Services Started Successfully!
echo ========================================
echo.
echo WEB INTERFACES:
echo   📊 Dashboard:     http://localhost:8501
echo   🔧 API docs:      http://localhost:8000/docs
echo   ❤️ Health check:  http://localhost:8000/health
echo   🗄️ Database:      http://localhost:5050 (optional)
echo.
echo WINDOWS OPENED:
echo   - RAG Server (port 8000)
echo   - Dashboard (port 8501)
echo.
echo TO STOP SERVICES:
echo   1. Close the server and dashboard command windows
echo   2. Run: docker-compose down
echo.
echo TROUBLESHOOTING:
echo   - If services don't start, check the opened command windows for errors
echo   - If ports are in use, restart your computer or kill processes manually
echo   - Check Docker Desktop is running and has enough resources
echo.

timeout /t 5 /nobreak >nul
echo Opening dashboard in your browser...
start http://localhost:8501

echo.
echo This setup window can be closed.
echo The application will continue running in the background.
echo.
pause