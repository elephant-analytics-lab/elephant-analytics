@echo off
setlocal

cd /d "%~dp0"

where docker >nul 2>nul
if errorlevel 1 (
  echo Docker Desktop is not installed or not in PATH.
  echo Please install Docker Desktop first, then run this file again.
  pause
  exit /b 1
)

docker compose version >nul 2>nul
if errorlevel 1 (
  echo Docker Compose is not available.
  echo Please open Docker Desktop and wait until it finishes starting, then run this file again.
  pause
  exit /b 1
)

echo Starting Elephant Analytics...
docker compose -f docker-compose.react.yml up --build -d
if errorlevel 1 (
  echo Failed to start the delivery package.
  echo Make sure Docker Desktop is running, then try again.
  pause
  exit /b 1
)

echo.
echo Services are starting.
echo React UI: http://127.0.0.1:8502
echo Backend API: http://127.0.0.1:8000
echo Tile server: http://127.0.0.1:8080
echo.
echo The first startup may take a few minutes because Docker needs to build images.
start "" "http://127.0.0.1:8502"
exit /b 0
