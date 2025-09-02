# quickstart.ps1
# PowerShell setup script for ARGO RAG PoC (more reliable than batch)

param(
    [switch]$SkipData,
    [switch]$Force
)

Write-Host ""
Write-Host "========================================" -ForegroundColor Blue
Write-Host "  ARGO RAG PoC - PowerShell Setup" -ForegroundColor Blue  
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Set error action preference
$ErrorActionPreference = "Stop"

try {
    # Check if we're in the right directory
    if (-not (Test-Path "config.py")) {
        Write-Host "ERROR: config.py not found" -ForegroundColor Red
        Write-Host "Please run this script from the project root directory" -ForegroundColor Red
        Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host "INFO: Working directory: $(Get-Location)" -ForegroundColor Green
    Write-Host ""

    # [1/7] Check Docker
    Write-Host "[1/7] Checking Docker..." -ForegroundColor Cyan
    try {
        $dockerVersion = docker --version 2>$null
        if ($LASTEXITCODE -ne 0) { throw "Docker not found" }
        
        # Check if Docker daemon is running
        docker ps 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker daemon not running" }
        
        Write-Host "SUCCESS: $dockerVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "ERROR: Docker is not available or not running" -ForegroundColor Red
        Write-Host "Please install Docker Desktop and make sure it's running" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    # [2/7] Check Python
    Write-Host "[2/7] Checking Python..." -ForegroundColor Cyan
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Python not found" }
        Write-Host "SUCCESS: $pythonVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "ERROR: Python is not available" -ForegroundColor Red
        Write-Host "Please install Python 3.10+ and add it to PATH" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    # [3/7] Setup virtual environment
    Write-Host "[3/7] Setting up Python environment..." -ForegroundColor Cyan
    if (-not (Test-Path "venv") -or $Force) {
        if (Test-Path "venv") {
            Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force "venv"
        }
        
        Write-Host "Creating virtual environment..." -ForegroundColor White
        python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
    }

    # Activate virtual environment and install packages
    Write-Host "Installing Python packages..." -ForegroundColor White
    & "venv\Scripts\Activate.ps1"
    
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Trying alternative installation..." -ForegroundColor Yellow
        python -m pip install --only-binary=all -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            throw "Package installation failed"
        }
    }
    Write-Host "SUCCESS: Python packages installed" -ForegroundColor Green

    # [4/7] Create configuration
    Write-Host "[4/7] Setting up configuration..." -ForegroundColor Cyan
    if (-not (Test-Path ".env") -or $Force) {
        Write-Host "Creating .env configuration file..." -ForegroundColor White
        @"
# ARGO RAG PoC Configuration
PG_USER=argo_user
PG_PASS=argo_pass
PG_DB=argo_db
PG_HOST=localhost
PG_PORT=5432
PARQUET_DIR=./parquet_store
CHROMA_DIR=./chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# Optional: Add your OpenAI API key below for enhanced responses
# OPENAI_API_KEY=your_key_here
"@ | Out-File -FilePath ".env" -Encoding UTF8
        Write-Host "SUCCESS: Configuration file created" -ForegroundColor Green
    } else {
        Write-Host "INFO: Configuration file already exists" -ForegroundColor Yellow
    }

    # [5/7] Setup database
    Write-Host "[5/7] Setting up database..." -ForegroundColor Cyan
    
    # Stop existing containers
    Write-Host "Stopping existing containers..." -ForegroundColor White
    docker-compose down --volumes 2>$null | Out-Null
    
    # Start PostgreSQL
    Write-Host "Starting PostgreSQL container..." -ForegroundColor White
    docker-compose up -d postgres
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start PostgreSQL container"
    }

    # Wait for PostgreSQL to be ready
    Write-Host "Waiting for PostgreSQL to initialize..." -ForegroundColor White
    $maxAttempts = 30
    $attempt = 0
    
    do {
        Start-Sleep -Seconds 2
        $attempt++
        Write-Host "  Attempt $attempt/$maxAttempts" -ForegroundColor Gray
        
        docker exec argo-postgres pg_isready -U argo_user -d argo_db 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: PostgreSQL is ready" -ForegroundColor Green
            break
        }
        
        if ($attempt -ge $maxAttempts) {
            throw "PostgreSQL failed to start within $($maxAttempts * 2) seconds"
        }
    } while ($true)

    # [6/7] Create and ingest sample data
    if (-not $SkipData) {
        Write-Host "[6/7] Setting up sample data..." -ForegroundColor Cyan
        
        if (-not (Test-Path "sample_data") -or $Force) {
            Write-Host "Creating sample ARGO data..." -ForegroundColor White
            python create_sample_data.py --output-dir sample_data --n-single 5 --n-multi 3
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to create sample data"
            }
            Write-Host "SUCCESS: Sample data created" -ForegroundColor Green
        }
        
        Write-Host "Ingesting sample data..." -ForegroundColor White
        python ingest.py --dir sample_data --parquet-dir parquet_store
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to ingest sample data"
        }
        Write-Host "SUCCESS: Sample data ingested" -ForegroundColor Green
    } else {
        Write-Host "[6/7] Skipping sample data creation" -ForegroundColor Yellow
    }

    # [7/7] Start services
    Write-Host "[7/7] Starting services..." -ForegroundColor Cyan
    
    # Kill existing processes on our ports
    Write-Host "Checking for existing processes..." -ForegroundColor White
    $processesToKill = Get-NetTCPConnection -LocalPort @(8000, 8501) -ErrorAction SilentlyContinue | Select-Object OwningProcess -Unique
    foreach ($proc in $processesToKill) {
        try {
            Stop-Process -Id $proc.OwningProcess -Force -ErrorAction SilentlyContinue
            Write-Host "  Stopped process $($proc.OwningProcess)" -ForegroundColor Gray
        } catch {}
    }
    
    Start-Sleep -Seconds 2
    
    # Start RAG server
    Write-Host "Starting RAG server..." -ForegroundColor White
    $serverJob = Start-Process -FilePath "cmd" -ArgumentList "/k", "cd /d `"$PWD`" && venv\Scripts\activate.bat && python -m uvicorn rag_server:app --reload --port 8000 --host 0.0.0.0" -WindowStyle Normal -PassThru
    
    # Wait for RAG server
    Write-Host "Waiting for RAG server to start..." -ForegroundColor White
    $maxAttempts = 15
    $attempt = 0
    
    do {
        Start-Sleep -Seconds 3
        $attempt++
        Write-Host "  Attempt $attempt/$maxAttempts" -ForegroundColor Gray
        
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Host "SUCCESS: RAG server is responding" -ForegroundColor Green
                break
            }
        }
        catch {
            # Continue waiting
        }
        
        if ($attempt -ge $maxAttempts) {
            Write-Host "WARNING: RAG server taking longer than expected" -ForegroundColor Yellow
            Write-Host "Continuing with dashboard startup..." -ForegroundColor Yellow
            break
        }
    } while ($true)
    
    # Start dashboard
    Write-Host "Starting Streamlit dashboard..." -ForegroundColor White
    $dashboardJob = Start-Process -FilePath "cmd" -ArgumentList "/k", "cd /d `"$PWD`" && venv\Scripts\activate.bat && streamlit run dashboard.py --server.port 8501" -WindowStyle Normal -PassThru

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "   SUCCESS: ARGO RAG PoC Started!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "WEB INTERFACES:" -ForegroundColor White
    Write-Host "   Dashboard:     http://localhost:8501" -ForegroundColor Cyan
    Write-Host "   API docs:      http://localhost:8000/docs" -ForegroundColor Cyan  
    Write-Host "   Health check:  http://localhost:8000/health" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "EXAMPLE QUERIES:" -ForegroundColor White
    Write-Host "   - 'temperature profiles in Atlantic Ocean'" -ForegroundColor Gray
    Write-Host "   - 'salinity data from 2023'" -ForegroundColor Gray
    Write-Host "   - 'profiles with oxygen measurements'" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "TO STOP SERVICES:" -ForegroundColor White
    Write-Host "   1. Close the server and dashboard windows" -ForegroundColor Gray
    Write-Host "   2. Run: docker-compose down" -ForegroundColor Gray
    Write-Host ""
    
    # Wait and open browser
    Write-Host "Opening dashboard in browser..." -ForegroundColor White
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:8501"
    
    Write-Host "Setup complete! Check the opened windows for any error messages." -ForegroundColor Green
    Read-Host "Press Enter to exit this setup script"

} catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "TROUBLESHOOTING:" -ForegroundColor Yellow
    Write-Host "  1. Make sure Docker Desktop is running" -ForegroundColor White
    Write-Host "  2. Check Python installation and PATH" -ForegroundColor White
    Write-Host "  3. Run as Administrator if permission issues" -ForegroundColor White
    Write-Host "  4. Try: .\reset_docker.bat to clean up containers" -ForegroundColor White
    Write-Host "  5. Run: .\diagnose.bat for detailed diagnosis" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}