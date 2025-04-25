#!/usr/bin/env python
"""
LangChain Import Update Script

This script helps update LangChain imports to the new package structure.
It uses the langchain CLI to automatically update imports in Python files.

Usage:
    python update_langchain_imports.py
"""

import os
import subprocess
import sys
from pathlib import Path

def check_langchain_cli():
    """Check if langchain CLI is installed."""
    try:
        subprocess.run(
            [sys.executable, "-m", "langchain", "--help"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_langchain_cli():
    """Install langchain CLI."""
    print("Installing langchain CLI...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "langchain-cli"],
        check=True
    )

def find_python_files(directory):
    """Find all Python files in the directory and its subdirectories."""
    return list(Path(directory).rglob("*.py"))

def update_imports(file_path):
    """Update imports in a Python file using langchain CLI."""
    print(f"Updating imports in {file_path}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "langchain", "migration", "upgrade", str(file_path)],
            capture_output=True,
            text=True,
            check=True
        )
        if "No changes made" in result.stdout:
            print(f"  No changes needed for {file_path}")
        else:
            print(f"  Updated imports in {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"  Error updating {file_path}: {e}")
        print(f"  STDOUT: {e.stdout}")
        print(f"  STDERR: {e.stderr}")

def main():
    """Main function to update LangChain imports."""
    print("LangChain Import Update Script")
    print("==============================")
    
    # Check if langchain CLI is installed
    if not check_langchain_cli():
        print("langchain CLI not found. Installing...")
        install_langchain_cli()
    
    # Get the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Scanning directory: {project_dir}")
    
    # Find all Python files
    python_files = find_python_files(project_dir)
    print(f"Found {len(python_files)} Python files")
    
    # Update imports in each file
    for file_path in python_files:
        update_imports(file_path)
    
    print("\nImport update completed.")
    print("Please review the changes and test your application.")

if __name__ == "__main__":
    main()