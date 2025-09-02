@echo off
setlocal EnableDelayedExpansion

REM reset_docker.bat
REM Script to completely reset Docker containers and PostgreSQL data

echo.
echo ========================================
echo  ARGO RAG PoC - Docker Reset Script
echo ========================================
echo.

cd /d "%~dp0"

echo WARNING: This will completely remove all Docker containers and database data!
echo.
set /p choice="Are you sure you want to continue? (y/N): "
if /i not "!choice!"=="y" if /i not "!choice!"=="yes" (
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo [1/5] Stopping all running containers...
docker-compose down --volumes --remove-orphans
if !errorlevel! neq 0 (
    echo Warning: Error stopping containers, continuing anyway...
)

echo [2/5] Removing containers and volumes...
docker-compose down --volumes --rmi local >nul 2>&1
docker container prune -f >nul 2>&1
docker volume prune -f >nul 2>&1

echo [3/5] Cleaning up local data directories...
if exist "chroma_db" (
    echo Removing ChromaDB data...
    rmdir /s /q "chroma_db" >nul 2>&1
)
if exist "parquet_store" (
    echo Removing Parquet files...
    rmdir /s /q "parquet_store" >nul 2>&1
)
if exist "sample_data" (
    echo Removing sample data...
    rmdir /s /q "sample_data" >nul 2>&1
)

echo [4/5] Rebuilding containers with fresh data...
docker-compose pull postgres
docker-compose up -d postgres --force-recreate

echo [5/5] Waiting for PostgreSQL to initialize...
set /a count=0
:wait_loop
timeout /t 3 /nobreak >nul
set /a count+=1

docker exec argo-postgres pg_isready -U argo_user -d argo_db >nul 2>&1
if !errorlevel! equ 0 (
    echo SUCCESS: PostgreSQL is ready with fresh database
    goto ready
)

if !count! gtr 20 (
    echo ERROR: PostgreSQL failed to start within 60 seconds
    docker-compose logs postgres
    pause
    exit /b 1
)

echo Attempt !count!/20: Waiting for PostgreSQL to initialize...
goto wait_loop

:ready

echo.
echo ========================================
echo   Docker Reset Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Create new sample data: python create_sample_data.py
echo   2. Ingest the data: python ingest.py --dir sample_data  
echo   3. Start services: start_services.bat
echo.
echo Or run the full setup: QUICKSTART_WINDOWS.bat
echo.
pause