@echo off
echo Starting local HTTP server for AidFlow AI...
echo.
echo This will allow you to load route files without CORS issues.
echo.
echo Once started, open your browser to:
echo http://localhost:8000/frontend/map.html
echo.
echo Press Ctrl+C to stop the server.
echo.
python -m http.server 8000
pause