@echo off
setlocal
cd /d %~dp0
if not exist .env (
  echo Missing .env file.
  echo Copy .env.example to .env and enter the camera RTSP URL privately.
  pause
  exit /b 1
)
if not exist .venv (
  py -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
