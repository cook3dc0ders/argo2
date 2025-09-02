# docker_reset.ps1
# Simple PowerShell script to reset Docker containers

Write-Host "ARGO RAG PoC - Docker Reset" -ForegroundColor Blue
Write-Host "==============================" -ForegroundColor Blue
Write-Host ""

$confirmation = Read-Host "This will remove all containers and data. Continue? (y/N)"
if ($confirmation -notmatch '^[Yy]') {
    Write-Host "Operation cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "Stopping containers..." -ForegroundColor Cyan
docker-compose down --volumes --remove-orphans

Write-Host "Removing images..." -ForegroundColor Cyan  
docker-compose down --rmi local 2>$null

Write-Host "Cleaning up volumes..." -ForegroundColor Cyan
docker volume prune -f

Write-Host "Removing local data..." -ForegroundColor Cyan
if (Test-Path "chroma_db") { Remove-Item -Recurse -Force "chroma_db" }
if (Test-Path "parquet_store") { Remove-Item -Recurse -Force "parquet_store" }
if (Test-Path "sample_data") { Remove-Item -Recurse -Force "sample_data" }

Write-Host ""
Write-Host "Starting fresh PostgreSQL container..." -ForegroundColor Cyan
docker-compose up -d postgres

Write-Host ""
Write-Host "Waiting for PostgreSQL to be ready..."
$maxAttempts = 30
for ($i = 1; $i -le $maxAttempts; $i++) {
    Start-Sleep -Seconds 2
    Write-Host "  Attempt $i/$maxAttempts" -ForegroundColor Gray
    
    $result = docker exec argo-postgres pg_isready -U argo_user -d argo_db 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: PostgreSQL is ready!" -ForegroundColor Green
        break
    }
    
    if ($i -eq $maxAttempts) {
        Write-Host "ERROR: PostgreSQL failed to start" -ForegroundColor Red
        Write-Host "Check logs with: docker-compose logs postgres"
        exit 1
    }
}

Write-Host ""
Write-Host "Reset complete! Next steps:" -ForegroundColor Green
Write-Host "  1. Run: .\quickstart.ps1"
Write-Host "  2. Or manually: python create_sample_data.py && python ingest.py --dir sample_data"