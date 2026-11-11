#!/bin/bash
# Quarry Maker - double-click this file to start (macOS / Linux).
# First start: installs what the program needs into a ".venv" folder here
# (a few minutes, needs internet). After that it starts in seconds.
cd "$(dirname "$0")" || exit 1

pause_exit() { echo; read -r -p "Press Enter to close this window." _; exit 1; }

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is needed."
  echo "Download it from https://www.python.org/downloads/ , install it, then double-click this file again."
  pause_exit
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "First start: preparing Quarry Maker (only once, a few minutes)..."
  "$PY" -m venv .venv || { echo "Could not create the Python environment."; pause_exit; }
fi
if ! .venv/bin/python -c "import rasterio, numpy, pyproj, shapely, PIL, yaml, requests" 2>/dev/null; then
  echo "Installing the parts Quarry Maker needs..."
  .venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1
  .venv/bin/python -m pip install -r requirements.txt || { echo "Installing failed. Check your internet connection and try again."; pause_exit; }
fi

.venv/bin/python app/quarry_app.py "$@"
