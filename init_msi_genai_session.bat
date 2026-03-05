@echo off
setlocal enabledelayedexpansion

chcp 65001 >nul 2>&1

set "WALKIE_DIR=%~dp0"
if not exist "%WALKIE_DIR%main_msi_genai.py" (
    echo [ERROR] Could not find main_msi_genai.py at:
    echo   %WALKIE_DIR%main_msi_genai.py
    exit /b 0
)

set "PYEXE=python"
if exist "%WALKIE_DIR%devenv\Scripts\python.exe" set "PYEXE=%WALKIE_DIR%devenv\Scripts\python.exe"

set "SESSION_FILE=%WALKIE_DIR%.msi_genai_session"

echo Initializing MSI GenAI session...
echo Output file: %SESSION_FILE%

"%PYEXE%" "%WALKIE_DIR%scripts\init_genai_session.py" "%SESSION_FILE%"

if exist "%SESSION_FILE%" (
    for /f "usebackq delims=" %%A in ("%SESSION_FILE%") do set "SID=%%A"
    echo.
    echo Saved session id: !SID!
    echo.
)

exit /b 0
