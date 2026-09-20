$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

Write-Host ''
Write-Host 'AutoGearUp CCTV - Secure Public Sharing' -ForegroundColor Cyan
Write-Host '---------------------------------------'

if (-not (Test-Path '.env')) {
    Write-Host 'Missing .env. Copy .env.example to .env and configure your cameras first.' -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}

if (-not (Test-Path '.venv')) { py -m venv .venv }
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt

$envText = Get-Content '.env' -Raw
$keyMatch = [regex]::Match($envText, '(?m)^REMOTE_ACCESS_KEY=(.+)$')
if ($keyMatch.Success -and $keyMatch.Groups[1].Value.Trim()) {
    $accessKey = $keyMatch.Groups[1].Value.Trim()
} else {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    $accessKey = [Convert]::ToHexString($bytes).ToLower()
    Add-Content '.env' ("`r`nREMOTE_ACCESS_KEY=" + $accessKey)
    Add-Content '.env' 'REMOTE_ALLOWED_ORIGIN=https://tanvirfuad.github.io'
    Write-Host 'Generated a new remote access key in .env.' -ForegroundColor Green
}

$cloudflared = Join-Path $PSScriptRoot 'cloudflared.exe'
if (-not (Test-Path $cloudflared)) {
    Write-Host 'Downloading Cloudflare Tunnel client...' -ForegroundColor Yellow
    Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile $cloudflared
}

Get-Process -Name 'cloudflared' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host 'Starting local CCTV gateway...'
$gateway = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','app:app','--host','127.0.0.1','--port','8000','--no-access-log' -WorkingDirectory $PSScriptRoot -PassThru
Start-Sleep -Seconds 2

Write-Host 'Starting authenticated public relay...'
$relay = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','public_proxy:app','--host','127.0.0.1','--port','8001','--no-access-log' -WorkingDirectory $PSScriptRoot -PassThru
Start-Sleep -Seconds 2

$outLog = Join-Path $PSScriptRoot 'cloudflared-out.log'
$errLog = Join-Path $PSScriptRoot 'cloudflared-err.log'
Remove-Item $outLog,$errLog -ErrorAction SilentlyContinue

Write-Host 'Opening encrypted Cloudflare tunnel...'
$tunnel = Start-Process -FilePath $cloudflared -ArgumentList 'tunnel','--url','http://127.0.0.1:8001','--no-autoupdate' -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru

$tunnelUrl = $null
for ($i=0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    $all = ''
    if (Test-Path $outLog) { $all += (Get-Content $outLog -Raw -ErrorAction SilentlyContinue) }
    if (Test-Path $errLog) { $all += (Get-Content $errLog -Raw -ErrorAction SilentlyContinue) }
    $m = [regex]::Match($all, 'https://[a-zA-Z0-9-]+\.trycloudflare\.com')
    if ($m.Success) { $tunnelUrl = $m.Value; break }
}

if (-not $tunnelUrl) {
    Write-Host ''
    Write-Host 'Tunnel started, but its URL could not be read automatically.' -ForegroundColor Yellow
    Write-Host 'Look in cloudflared-err.log for the trycloudflare.com URL.'
    Read-Host 'Press Enter to close'
    exit 1
}

$encodedGateway = [System.Uri]::EscapeDataString($tunnelUrl)
$encodedKey = [System.Uri]::EscapeDataString($accessKey)
$dashboard = 'https://tanvirfuad.github.io/autogearup/cctv-analytics-demo/#gateway=' + $encodedGateway + '&key=' + $encodedKey

Write-Host ''
Write-Host 'PUBLIC CCTV DASHBOARD IS READY' -ForegroundColor Green
Write-Host ''
Write-Host $dashboard -ForegroundColor Cyan
Write-Host ''
Write-Host 'The RTSP passwords are NOT in this link.' -ForegroundColor Green
Write-Host 'The access key is stored in the URL fragment and then browser local storage.'
Write-Host 'Keep this window open while you want public viewing to work.'
Write-Host ''
Set-Clipboard -Value $dashboard
Write-Host 'The secure dashboard link has been copied to your clipboard.'
Start-Process $dashboard

Read-Host 'Press Enter to stop public sharing'
Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $relay.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $gateway.Id -Force -ErrorAction SilentlyContinue
