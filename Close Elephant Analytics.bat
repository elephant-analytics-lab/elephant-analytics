@echo off
setlocal

cd /d "%~dp0"

where docker >nul 2>nul
if errorlevel 1 (
  echo Docker Desktop is not installed or not in PATH.
  pause
  exit /b 1
)

docker compose version >nul 2>nul
if errorlevel 1 (
  echo Docker Compose is not available.
  echo Please open Docker Desktop and try again.
  pause
  exit /b 1
)

echo Stopping Elephant Analytics...
docker compose -f docker-compose.react.yml down
if errorlevel 1 (
  echo Failed to stop containers cleanly.
  pause
  exit /b 1
)

echo Delivery package stopped.
exit /b 0
