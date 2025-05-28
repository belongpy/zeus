@echo off
setlocal enabledelayedexpansion

REM =====================================================================
REM Zeus Wallet Analysis System - Launcher
REM =====================================================================
REM Quick launcher for Zeus CLI
REM Usage: Double-click start.bat or run from command prompt
REM =====================================================================

title Zeus - Wallet Analysis System
color 0B

echo.
echo ======================================================================
echo                    ⚡ ZEUS LAUNCHER ⚡
echo              Standalone Wallet Analysis System
echo ======================================================================
echo.

REM Check if we're in the zeus directory
if not exist "zeus_cli.py" (
    echo ❌ Error: zeus_cli.py not found!
    echo 📁 Make sure you're running this from the Zeus directory
    echo 💡 Expected files: zeus_cli.py, zeus_analyzer.py, etc.
    echo.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found!
    echo 📥 Please install Python from https://python.org
    echo 💡 Make sure to check "Add to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if requirements are installed
echo.
echo 🔍 Checking dependencies...
python -c "import requests, numpy, base58" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Missing dependencies detected
    echo 📦 Installing requirements...
    echo.
    
    if exist "requirements.txt" (
        pip install -r requirements.txt
        if errorlevel 1 (
            echo ❌ Failed to install dependencies
            echo 💡 Try: pip install requests numpy pandas base58
            echo.
            pause
            exit /b 1
        )
        echo ✅ Dependencies installed successfully
    ) else (
        echo ❌ requirements.txt not found
        echo 📦 Installing basic dependencies...
        pip install requests numpy pandas base58
    )
) else (
    echo ✅ Dependencies OK
)

echo.
echo 🚀 Starting Zeus...
echo ======================================================================
echo.

REM Launch Zeus CLI
python zeus_cli.py

REM Check if Zeus exited with error
if errorlevel 1 (
    echo.
    echo ======================================================================
    echo ❌ Zeus exited with error
    echo 💡 Check the error messages above
    echo 🔧 Try: python zeus_cli.py configure
    echo.
) else (
    echo.
    echo ======================================================================
    echo ✅ Zeus session completed successfully
    echo.
)

pause