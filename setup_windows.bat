@echo off

REM Automated Enterprise Setup Script for Windows
setlocal enabledelayedexpansion

REM Check Python version
python --version 3>nul
if %errorlevel% neq 0 (
    echo [ERROR] [%DATE% %TIME%] Python 3.8+ required | tee -a setup.log
    exit /b 101
)

REM Check for required binaries
where openssl >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] [%DATE% %TIME%] OpenSSL not found | tee -a setup.log
    exit /b 102
)

where mysql >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] [%DATE% %TIME%] MySQL client not installed | tee -a setup.log
    exit /b 103
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv --copies
if %errorlevel% neq 0 (
    echo [ERROR] [%DATE% %TIME%] Failed to create isolated virtual environment | tee -a setup.log
    exit /b 104
)
if %errorlevel% neq 0 (
    echo Error creating virtual environment
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
call .\venv\Scripts\activate
pip install --require-virtualenv --no-cache-dir -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] [%DATE% %TIME%] Dependency installation failed | tee -a setup.log
    exit /b 105
)

REM Verify cryptographic libraries
python -c "import cryptography; print('Cryptography version:', cryptography.__version__)"
if %errorlevel% neq 0 (
    echo [ERROR] [%DATE% %TIME%] Cryptographic dependencies missing | tee -a setup.log
    exit /b 106
)
if %errorlevel% neq 0 (
    echo Error installing dependencies
    exit /b 1
)

REM Database setup
echo Initializing database...
mysql --ssl-mode=REQUIRED -e "CREATE DATABASE IF NOT EXISTS candidate_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
if %errorlevel% neq 0 (
    echo [ERROR] [%DATE% %TIME%] Secure database creation failed | tee -a setup.log
    exit /b 107
)
if %errorlevel% neq 0 (
    echo Error creating database
    exit /b 1
)

REM Create uploads directory
echo Creating directories...
mkdir uploads 2>nul

REM Environment checks
echo Running system checks...
python -c "import sys; from src.utils.setup import validate_environment; validate_environment()"
if %errorlevel% neq 0 (
    echo Environment validation failed
    exit /b 1
)

echo [SUCCESS] [%DATE% %TIME%] Enterprise setup completed

REM Run security audit
python -m venv\Scripts\python -c "from src.tools.security_middleware import run_security_audit; run_security_audit()"
if %errorlevel% neq 0 (
    echo [WARNING] [%DATE% %TIME%] Security audit found configuration issues | tee -a setup.log
)

REM Generate setup report
python -m venv\Scripts\python -c "from src.utils.reporting import generate_setup_report; generate_setup_report()"
pause