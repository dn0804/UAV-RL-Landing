# 1. Define the possible path for VcXsrv
$vcxsrvPath = "C:\Program Files\VcXsrv\vcxsrv.exe"

# 2. Check/Install VcXsrv
if (-not (Test-Path $vcxsrvPath) -and -not (Get-Command "vcxsrv" -ErrorAction SilentlyContinue)) {
    Write-Host "VcXsrv not found. Installing via Winget..." -ForegroundColor Cyan
    winget install marha.VcXsrv --accept-package-agreements --accept-source-agreements
}

# 3. Launch VcXsrv if it's NOT RUNNING
$process = Get-Process "vcxsrv" -ErrorAction SilentlyContinue
if ($null -eq $process) {
    Write-Host "VcXsrv is not running. Launching automatically..." -ForegroundColor Yellow
    
    $xArgs = "-multiwindow -clipboard -wgl -ac"
    
    # Try to start using the full path first, then fall back to the command name
    if (Test-Path $vcxsrvPath) {
        Start-Process $vcxsrvPath -ArgumentList $xArgs -WindowStyle Hidden
    } else {
        # This is a fallback in case it was installed in a custom location
        Start-Process "vcxsrv.exe" -ArgumentList $xArgs -WindowStyle Hidden
    }
    
    Start-Sleep -Seconds 2
    Write-Host "VcXsrv started successfully." -ForegroundColor Green
} else {
    Write-Host "VcXsrv is already running." -ForegroundColor Gray
}

# 4. Launch Docker
$env:DISPLAY = "host.docker.internal:0.0"
Write-Host "Starting Docker..." -ForegroundColor Green
docker compose up -d --build

Write-Host "Container started! Run 'docker exec -it uav_rl_container bash' to enter." -ForegroundColor Cyan