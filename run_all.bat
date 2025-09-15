@echo off
echo Starting all 5 Flask apps with Waitress...

:: Main entry (links page)
start cmd /k "waitress-serve --listen=0.0.0.0:8000 app:app"

:: Other apps
start cmd /k "waitress-serve --listen=0.0.0.0:8001 app1:app"
start cmd /k "waitress-serve --listen=0.0.0.0:8002 app2:app"
start cmd /k "waitress-serve --listen=0.0.0.0:8003 app3:app"
start cmd /k "waitress-serve --listen=0.0.0.0:8004 app4:app"

echo All apps started!
pause
