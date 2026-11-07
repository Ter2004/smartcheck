@echo off
rem Test / demo day: SmartCheck server with request timing (Server-Timing + log)
rem and the Cloudflare tunnel pointed at 127.0.0.1 (localhost costs ~205 ms per new
rem connection on this Windows PC). Each runs in its own window; close a window to stop it.
setlocal
cd /d "%~dp0..\.."
if not exist "venv\Scripts\python.exe" (
    echo SmartCheck venv not found: %CD%\venv\Scripts\python.exe
    pause
    exit /b 1
)
start "SmartCheck server" cmd /k "set PERFORMANCE_LOG_ENABLED=true&& set FLASK_DEBUG=0&& venv\Scripts\python.exe run.py"

set CF=cloudflared
if exist "cloudflared.exe" set CF=cloudflared.exe
if exist "%USERPROFILE%\.cloudflared\config.yml" (
    start "SmartCheck tunnel" cmd /k "%CF% tunnel run smartcheck"
    echo Server and named tunnel started.
) else (
    echo [!] No named tunnel yet - see docs\named-tunnel.md
    echo     Starting a QUICK tunnel instead: its address changes every time,
    echo     so the Android app must be rebuilt with the new address.
    start "SmartCheck tunnel (quick)" cmd /k "%CF% tunnel --url http://127.0.0.1:5000"
)
pause
