$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Pause-And-Exit([string]$Message, [int]$Code = 1) {
    Write-Host ''
    Write-Host $Message -ForegroundColor Red
    Write-Host ''
    Read-Host 'Press Enter to close'
    exit $Code
}

try {
    Write-Host ''
    Write-Host 'AutoGearUp CCTV - Secure Public Sharing' -ForegroundColor Cyan
    Write-Host '---------------------------------------'
    Write-Host ('Folder: ' + $PSScriptRoot) -ForegroundColor DarkGray
    Write-Host ''

    if (-not (Test-Path '.env')) {
        Pause-And-Exit 'Missing .env. Copy .env.example to .env and configure your cameras first.'
    }

    $pyCommand = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pyCommand = 'py'
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pyCommand = 'python'
    } else {
        Pause-And-Exit 'Python was not found. Install Python 3.11 or 3.12 and enable Add Python to PATH.'
    }

    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        Write-Host 'Creating Python virtual environment...' -ForegroundColor Yellow
        & $pyCommand -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
    }

    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $python)) { throw 'Virtual-environment Python was not created correctly.' }

    Write-Host 'Checking Python packages...' -ForegroundColor Yellow
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Python package installation failed.' }

    $envText = Get-Content '.env' -Raw
    $keyMatch = [regex]::Match($envText, '(?m)^REMOTE_ACCESS_KEY=(.+)$')
    if ($keyMatch.Success -and $keyMatch.Groups[1].Value.Trim()) {
        $accessKey = $keyMatch.Groups[1].Value.Trim()
    } else {
        Write-Host 'Creating secure remote access key...' -ForegroundColor Yellow
        $bytes = New-Object byte[] 32
        $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        try {
            $rng.GetBytes($bytes)
        } finally {
            $rng.Dispose()
        }
        $accessKey = -join ($bytes | ForEach-Object { $_.ToString('x2') })
        Add-Content '.env' ("`r`nREMOTE_ACCESS_KEY=" + $accessKey)
        if ($envText -notmatch '(?m)^REMOTE_ALLOWED_ORIGIN=') {
            Add-Content '.env' 'REMOTE_ALLOWED_ORIGIN=https://tanvirfuad.github.io'
        }
        Write-Host 'Secure access key created.' -ForegroundColor Green
    }

    $cloudflared = Join-Path $PSScriptRoot 'cloudflared.exe'
    if (-not (Test-Path $cloudflared)) {
        Write-Host 'Downloading Cloudflare Tunnel client...' -ForegroundColor Yellow
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile $cloudflared
        } catch {
            throw ('Could not download cloudflared.exe: ' + $_.Exception.Message)
        }
    }

    Get-Process -Name 'cloudflared' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

    $gatewayLog = Join-Path $PSScriptRoot 'gateway.log'
    $relayLog = Join-Path $PSScriptRoot 'relay.log'
    $gatewayErr = Join-Path $PSScriptRoot 'gateway-error.log'
    $relayErr = Join-Path $PSScriptRoot 'relay-error.log'
    Remove-Item $gatewayLog,$relayLog,$gatewayErr,$relayErr -ErrorAction SilentlyContinue

    Write-Host 'Starting local CCTV gateway...' -ForegroundColor Yellow
    $gateway = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','app:app','--host','127.0.0.1','--port','8000','--no-access-log' -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $gatewayLog -RedirectStandardError $gatewayErr -PassThru
    Start-Sleep -Seconds 3
    if ($gateway.HasExited) {
        $msg = ''
        if (Test-Path $gatewayErr) { $msg = Get-Content $gatewayErr -Raw -ErrorAction SilentlyContinue }
        throw ('Local CCTV gateway failed to start. ' + $msg)
    }

    Write-Host ''
    Write-Host 'Checking AI camera status...' -ForegroundColor Yellow
    try {
        $health = Invoke-RestMethod -UseBasicParsing -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 15
        foreach ($cam in $health.cameras) {
            $ai = if ($cam.analytics_enabled) { 'AI ON' } else { 'AI OFF' }
            $online = if ($cam.online) { 'ONLINE' } else { ($cam.status).ToUpper() }
            Write-Host ('Camera ' + $cam.id + ' - ' + $cam.name + ': ' + $online + ' | ' + $ai + ' | Now: ' + $cam.current_people + ' people / ' + $cam.current_vehicles + ' vehicles')
        }
        if (-not $health.analytics_enabled) {
            Write-Host ''
            Write-Host 'WARNING: Video may work, but YOLO AI is not loaded.' -ForegroundColor Red
            Write-Host 'Check gateway-error.log and gateway.log in this folder.' -ForegroundColor Yellow
        }
    } catch {
        Write-Host ('Could not read AI health yet: ' + $_.Exception.Message) -ForegroundColor Yellow
    }
    Write-Host ''
    Write-Host 'Starting authenticated public relay...' -ForegroundColor Yellow
    $relay = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','public_proxy:app','--host','127.0.0.1','--port','8001','--no-access-log' -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $relayLog -RedirectStandardError $relayErr -PassThru
    Start-Sleep -Seconds 3
    if ($relay.HasExited) {
        $msg = ''
        if (Test-Path $relayErr) { $msg = Get-Content $relayErr -Raw -ErrorAction SilentlyContinue }
        throw ('Public relay failed to start. ' + $msg)
    }

    $outLog = Join-Path $PSScriptRoot 'cloudflared-out.log'
    $errLog = Join-Path $PSScriptRoot 'cloudflared-err.log'
    Remove-Item $outLog,$errLog -ErrorAction SilentlyContinue

    Write-Host 'Opening encrypted Cloudflare tunnel...' -ForegroundColor Yellow
    $tunnel = Start-Process -FilePath $cloudflared -ArgumentList 'tunnel','--url','http://127.0.0.1:8001','--no-autoupdate' -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru

    $tunnelUrl = $null
    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 1
        if ($tunnel.HasExited) { break }
        $all = ''
        if (Test-Path $outLog) { $all += (Get-Content $outLog -Raw -ErrorAction SilentlyContinue) }
        if (Test-Path $errLog) { $all += (Get-Content $errLog -Raw -ErrorAction SilentlyContinue) }
        $m = [regex]::Match($all, 'https://[a-zA-Z0-9-]+\.trycloudflare\.com')
        if ($m.Success) {
            $tunnelUrl = $m.Value
            break
        }
        Write-Host -NoNewline '.'
    }
    Write-Host ''

    if (-not $tunnelUrl) {
        $details = ''
        if (Test-Path $errLog) { $details = Get-Content $errLog -Raw -ErrorAction SilentlyContinue }
        throw ('Cloudflare tunnel did not return a public URL. ' + $details)
    }

    $encodedGateway = [System.Uri]::EscapeDataString($tunnelUrl)
    $encodedKey = [System.Uri]::EscapeDataString($accessKey)
    $dashboard = 'https://tanvirfuad.github.io/autogearup/cctv-analytics-demo/#gateway=' + $encodedGateway + '&key=' + $encodedKey

    Write-Host ''
    Write-Host 'PUBLIC CCTV DASHBOARD IS READY' -ForegroundColor Green
    Write-Host ''
    Write-Host ('Tunnel: ' + $tunnelUrl) -ForegroundColor DarkGray
    Write-Host ''
    Write-Host $dashboard -ForegroundColor Cyan
    Write-Host ''
    Write-Host 'The RTSP usernames/passwords are NOT in this link.' -ForegroundColor Green
    Write-Host 'Keep this window open while remote viewing is enabled.'
    Write-Host ''

    try { Set-Clipboard -Value $dashboard } catch { }
    $linkFile = Join-Path $PSScriptRoot 'CURRENT_PUBLIC_LINK.txt'
    Set-Content -Path $linkFile -Value $dashboard -Encoding UTF8
    Write-Host 'The secure dashboard link has been copied to your clipboard.'
    Write-Host 'It is also saved as CURRENT_PUBLIC_LINK.txt in this folder.' -ForegroundColor Green
    Write-Host 'IMPORTANT: older trycloudflare.com links stop working after a restart.' -ForegroundColor Yellow
    Start-Process $dashboard

    Read-Host 'Press Enter to stop public sharing'

    Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $relay.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $gateway.Id -Force -ErrorAction SilentlyContinue
}
catch {
    Write-Host ''
    Write-Host 'STARTUP ERROR' -ForegroundColor Red
    Write-Host '-------------' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if ($_.InvocationInfo -and $_.InvocationInfo.PositionMessage) {
        Write-Host ''
        Write-Host $_.InvocationInfo.PositionMessage -ForegroundColor DarkYellow
    }
    Write-Host ''
    Write-Host 'Useful log files are in this same gateway folder:' -ForegroundColor Yellow
    Write-Host 'gateway-error.log, relay-error.log, cloudflared-err.log'
    Write-Host ''
    Read-Host 'Press Enter to close'
    exit 1
}
