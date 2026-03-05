@echo off
REM ========================================
REM Walkie-Tracker MSI GenAI Runner
REM Runs: ..\walkie-tracker\main_msi_genai.py
REM ========================================

setlocal enabledelayedexpansion

chcp 65001 >nul 2>&1

REM Resolve walkie-tracker folder (sibling of this folder)
set "WALKIE_DIR=%~dp0..\walkie-tracker"

if not exist "%WALKIE_DIR%\main_msi_genai.py" (
    echo [ERROR] Could not find main_msi_genai.py at:
    echo   %WALKIE_DIR%\main_msi_genai.py
    echo.
    echo Update WALKIE_DIR in this .bat if your folder is elsewhere.
    exit /b 0
)

REM Optional: close Windows Camera app (exclusive access)
tasklist /FI "IMAGENAME eq WindowsCamera.exe" 2>NUL | find /I /N "WindowsCamera.exe">NUL
if %ERRORLEVEL% equ 0 (
    taskkill /IM WindowsCamera.exe /F >NUL 2>&1
    timeout /t 2 /nobreak > NUL
)

cd /d "%WALKIE_DIR%"

echo Running Walkie-Tracker MSI GenAI...
echo Workdir: %CD%

set "PYEXE=python"
if exist "%WALKIE_DIR%\devenv\Scripts\python.exe" set "PYEXE=%WALKIE_DIR%\devenv\Scripts\python.exe"

if "%WALKIE_CLI_ARGS%"=="" (
    "%PYEXE%" main_msi_genai.py --gui
) else (
    "%PYEXE%" main_msi_genai.py %WALKIE_CLI_ARGS%
)
set "PY_EXIT=%errorlevel%"
if not "%PY_EXIT%"=="0" (
    echo [INFO] Python exited with code %PY_EXIT% (ignored; returning 0)
)
exit /b 0
