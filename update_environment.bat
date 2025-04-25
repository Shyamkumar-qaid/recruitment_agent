@echo off
echo ===================================
echo AI Resume Analyzer - Environment Update
echo ===================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python and try again.
    exit /b 1
)

echo Updating dependencies...
echo.

REM Update pip
python -m pip install --upgrade pip

REM Install dependencies with specific versions
python -m pip install pyasn1>=0.6.1,<0.7.0
python -m pip install -r requirements.txt

REM Run dependency resolution script if conflicts persist
python update_dependencies.py

echo.
echo Updating LangChain imports...
echo.

REM Install langchain-cli if needed and update imports
python -m pip install langchain-cli
python update_langchain_imports.py

echo.
echo Environment update completed.
echo.
echo If you still encounter issues, please try:
echo 1. Creating a fresh virtual environment
echo 2. Installing dependencies in the new environment
echo.
echo python -m venv venv
echo venv\Scripts\activate
echo pip install -r requirements.txt
echo python update_langchain_imports.py
echo.
pause