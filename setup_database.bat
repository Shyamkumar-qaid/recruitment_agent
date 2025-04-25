@echo off
echo ===================================
echo AI Resume Analyzer - Database Setup
echo ===================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python and try again.
    exit /b 1
)

echo Choose database type:
echo 1. SQLite (recommended for development)
echo 2. MySQL
echo.
set /p DB_TYPE="Enter your choice (1 or 2): "

if "%DB_TYPE%"=="1" (
    echo.
    echo Setting up SQLite database...
    python setup_database.py --type sqlite
) else if "%DB_TYPE%"=="2" (
    echo.
    echo Setting up MySQL database...
    
    set /p DB_USER="Enter MySQL username (default: root): "
    if "%DB_USER%"=="" set DB_USER=root
    
    set /p DB_PASSWORD="Enter MySQL password: "
    
    set /p DB_HOST="Enter MySQL host (default: localhost): "
    if "%DB_HOST%"=="" set DB_HOST=localhost
    
    set /p DB_PORT="Enter MySQL port (default: 3306): "
    if "%DB_PORT%"=="" set DB_PORT=3306
    
    set /p DB_NAME="Enter database name (default: candidate_db): "
    if "%DB_NAME%"=="" set DB_NAME=candidate_db
    
    python setup_database.py --type mysql --user "%DB_USER%" --password "%DB_PASSWORD%" --host "%DB_HOST%" --port "%DB_PORT%" --name "%DB_NAME%"
) else (
    echo Invalid choice. Please run the script again and select 1 or 2.
    exit /b 1
)

echo.
echo Database setup completed.
echo.
pause