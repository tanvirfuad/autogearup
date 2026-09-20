@echo off
title AutoGearUp CCTV Public Sharing
cd /d "%~dp0"
echo Starting AutoGearUp CCTV public sharing...
echo.
powershell.exe -NoLogo -NoProfile -NoExit -ExecutionPolicy Bypass -File "%~dp0start-public-windows.ps1"
echo.
echo PowerShell exited.
pause
