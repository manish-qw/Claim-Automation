# ===============================================================
#  CLAIMOS - Full Stack Startup Script
#  Starts: PostgreSQL + Redis + Neo4j (Docker) + FastAPI Server
# ===============================================================

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "   CLAIMOS AI - Fraud Intelligence Pipeline          " -ForegroundColor Cyan
Write-Host "   Hackathon Edition                                 " -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1 - Start Docker services
Write-Host "[*] Starting Docker services (postgres, redis, neo4j)..." -ForegroundColor Yellow
Set-Location "$PSScriptRoot\infrastructure"
docker compose up -d postgres redis neo4j
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Docker failed. Make sure Docker Desktop is running." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Docker services started" -ForegroundColor Green

# Step 2 - Wait for PostgreSQL to be ready
Write-Host "[*] Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
$retries = 10
for ($i = 1; $i -le $retries; $i++) {
    $result = docker exec claimos_postgres pg_isready -U postgres
    if ($result -match "accepting connections") {
        Write-Host "[OK] PostgreSQL ready" -ForegroundColor Green
        break
    }
    Write-Host "  Waiting... ($i/$retries)"
    Start-Sleep -Seconds 2
}

# Step 3 - Start FastAPI server
Write-Host ""
Write-Host "[*] Starting FastAPI server on http://localhost:8001 ..." -ForegroundColor Yellow
Write-Host "  -> API Docs : http://localhost:8001/docs" -ForegroundColor Cyan
Write-Host "  -> Health   : http://localhost:8001/fraud/health" -ForegroundColor Cyan
Write-Host "  -> Results  : http://localhost:8001/fraud/results" -ForegroundColor Cyan
Write-Host "  -> Analyze  : POST http://localhost:8001/fraud/analyze" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Press CTRL+C to stop the server." -ForegroundColor Gray
Write-Host ""

Set-Location "$PSScriptRoot"
uvicorn fraud_pipeline.main:app --host 0.0.0.0 --port 8001 --reload
