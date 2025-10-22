@echo off
REM Quick Start Script for Digital Restoration of Paintings (Windows)
REM This script sets up the environment and runs the application

echo ==========================================
echo Digital Restoration of Paintings
echo Quick Start Script
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed.
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo. Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo. Virtual environment created
) else (
    echo. Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo. Virtual environment activated
echo.

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo. Dependencies installed successfully
) else (
    echo. Error installing dependencies
    pause
    exit /b 1
)
echo.

REM Create necessary directories
echo Creating directories...
if not exist "sample_images" mkdir sample_images
if not exist "test_outputs" mkdir test_outputs
echo. Directories created
echo.

REM Ask to run tests
set /p runtests="Run algorithm tests? (y/n): "
if /i "%runtests%"=="y" (
    echo Running tests...
    python test_algorithms.py
    echo.
)

REM Start the application
echo ==========================================
echo Starting Streamlit application...
echo ==========================================
echo.
echo The application will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py


