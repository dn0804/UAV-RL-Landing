# 1. Check/Install VcXsrv
if (-not (Get-Command "vcxsrv" -ErrorAction SilentlyContinue)) {
    Write-Host "VcXsrv not found. Installing via Winget..." -ForegroundColor Cyan
    winget install marha.VcXsrv
    Write-Host "Installation complete. Please launch 'XLaunch' from the Start Menu." -ForegroundColor Green
    Write-Host "   (Settings: Multiple Windows -> Start no client -> CHECK 'Disable Access Control')" -ForegroundColor Yellow
}

# 2. Check if VcXsrv is actually RUNNING
$process = Get-Process "vcxsrv" -ErrorAction SilentlyContinue
if ($null -eq $process) {
    Write-Host "VcXsrv is installed but NOT RUNNING." -ForegroundColor Red
    Write-Host "Please start XLaunch with 'Disable Access Control' checked."
    exit
}

# 3. Launch Docker
$env:DISPLAY = "host.docker.internal:0.0"
Write-Host "Starting Docker..." -ForegroundColor Green
docker compose up -d --build
Write-Host "Container started! Run 'docker exec -it uav_rl_container bash' to enter."