@echo off
rem Quarry Maker - double-click this file to start (Windows).
rem First start: installs what the program needs into a ".venv" folder here
rem (a few minutes, needs internet). After that it starts in seconds.
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY goto nopython
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul || goto nopython

if not exist ".venv\Scripts\python.exe" (
  echo First start: preparing Quarry Maker ^(only once, a few minutes^)...
  %PY% -m venv .venv || goto failed
)
".venv\Scripts\python.exe" -c "import rasterio, numpy, pyproj, shapely, PIL, yaml, requests" >nul 2>nul
if errorlevel 1 (
  echo Installing the parts Quarry Maker needs...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto failed
)

".venv\Scripts\python.exe" app\quarry_app.py %*
pause
exit /b 0

:nopython
echo Python 3.10 or newer is needed.
echo Download it from https://www.python.org/downloads/ and install it.
echo IMPORTANT: on the first installer screen, tick "Add python.exe to PATH".
echo Then double-click this file again.
pause
exit /b 1

:failed
echo Installing failed. Check your internet connection and try again.
pause
exit /b 1
