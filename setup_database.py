#!/usr/bin/env python
"""
Database Setup Script

This script helps set up the database for the AI Resume Analyzer.
It creates the necessary tables and initializes the database.

Usage:
    python setup_database.py
"""

import os
import sys
import argparse
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from models import Base, Candidate, AuditLog

def setup_sqlite_database(db_name="candidate_db.db"):
    """Set up a SQLite database for development."""
    try:
        # Create SQLite database
        engine = create_engine(f"sqlite:///{db_name}")
        
        # Create tables
        Base.metadata.create_all(engine)
        
        # Check if tables were created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if 'candidates' in tables and 'audit_logs' in tables:
            print(f"SQLite database '{db_name}' set up successfully!")
            print(f"Tables created: {', '.join(tables)}")
            
            # Update .env file to use SQLite
            update_env_for_sqlite(db_name)
            
            return True
        else:
            print("Error: Not all tables were created.")
            return False
            
    except SQLAlchemyError as e:
        print(f"Error setting up SQLite database: {str(e)}")
        return False

def setup_mysql_database(user, password, host, port, db_name):
    """Set up a MySQL database."""
    try:
        # Create MySQL database connection
        engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}")
        
        # Create database if it doesn't exist
        with engine.connect() as conn:
            conn.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            conn.execute(f"USE {db_name}")
        
        # Connect to the specific database
        db_engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}")
        
        # Create tables
        Base.metadata.create_all(db_engine)
        
        # Check if tables were created
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        
        if 'candidates' in tables and 'audit_logs' in tables:
            print(f"MySQL database '{db_name}' set up successfully!")
            print(f"Tables created: {', '.join(tables)}")
            
            # Update .env file to use MySQL
            update_env_for_mysql(user, password, host, port, db_name)
            
            return True
        else:
            print("Error: Not all tables were created.")
            return False
            
    except SQLAlchemyError as e:
        print(f"Error setting up MySQL database: {str(e)}")
        return False

def update_env_for_sqlite(db_name):
    """Update .env file to use SQLite."""
    env_file = ".env"
    
    # Create .env file if it doesn't exist
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("# Database Configuration\n")
            f.write(f"DB_TYPE=sqlite\n")
            f.write(f"DB_NAME={db_name}\n\n")
            f.write("# LLM Configuration\n")
            f.write("OLLAMA_MODEL=phi3\n")
            f.write("OLLAMA_BASE_URL=http://localhost:11434\n")
        print(f"Created new .env file with SQLite configuration.")
        return
    
    # Read existing .env file
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    # Update or add SQLite configuration
    db_type_found = False
    db_name_found = False
    
    for i, line in enumerate(lines):
        if line.startswith("DB_TYPE="):
            lines[i] = f"DB_TYPE=sqlite\n"
            db_type_found = True
        elif line.startswith("DB_NAME="):
            lines[i] = f"DB_NAME={db_name}\n"
            db_name_found = True
    
    if not db_type_found:
        lines.append(f"DB_TYPE=sqlite\n")
    if not db_name_found:
        lines.append(f"DB_NAME={db_name}\n")
    
    # Write updated .env file
    with open(env_file, "w") as f:
        f.writelines(lines)
    
    print(f"Updated .env file with SQLite configuration.")

def update_env_for_mysql(user, password, host, port, db_name):
    """Update .env file to use MySQL."""
    env_file = ".env"
    
    # Create .env file if it doesn't exist
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write("# Database Configuration\n")
            f.write(f"DB_TYPE=mysql\n")
            f.write(f"DB_USER={user}\n")
            f.write(f"DB_PASSWORD={password}\n")
            f.write(f"DB_HOST={host}\n")
            f.write(f"DB_PORT={port}\n")
            f.write(f"DB_NAME={db_name}\n\n")
            f.write("# LLM Configuration\n")
            f.write("OLLAMA_MODEL=phi3\n")
            f.write("OLLAMA_BASE_URL=http://localhost:11434\n")
        print(f"Created new .env file with MySQL configuration.")
        return
    
    # Read existing .env file
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    # Update or add MySQL configuration
    db_type_found = False
    db_user_found = False
    db_password_found = False
    db_host_found = False
    db_port_found = False
    db_name_found = False
    
    for i, line in enumerate(lines):
        if line.startswith("DB_TYPE="):
            lines[i] = f"DB_TYPE=mysql\n"
            db_type_found = True
        elif line.startswith("DB_USER="):
            lines[i] = f"DB_USER={user}\n"
            db_user_found = True
        elif line.startswith("DB_PASSWORD="):
            lines[i] = f"DB_PASSWORD={password}\n"
            db_password_found = True
        elif line.startswith("DB_HOST="):
            lines[i] = f"DB_HOST={host}\n"
            db_host_found = True
        elif line.startswith("DB_PORT="):
            lines[i] = f"DB_PORT={port}\n"
            db_port_found = True
        elif line.startswith("DB_NAME="):
            lines[i] = f"DB_NAME={db_name}\n"
            db_name_found = True
    
    if not db_type_found:
        lines.append(f"DB_TYPE=mysql\n")
    if not db_user_found:
        lines.append(f"DB_USER={user}\n")
    if not db_password_found:
        lines.append(f"DB_PASSWORD={password}\n")
    if not db_host_found:
        lines.append(f"DB_HOST={host}\n")
    if not db_port_found:
        lines.append(f"DB_PORT={port}\n")
    if not db_name_found:
        lines.append(f"DB_NAME={db_name}\n")
    
    # Write updated .env file
    with open(env_file, "w") as f:
        f.writelines(lines)
    
    print(f"Updated .env file with MySQL configuration.")

def main():
    """Main function to set up the database."""
    parser = argparse.ArgumentParser(description="Set up the database for AI Resume Analyzer")
    parser.add_argument("--type", choices=["sqlite", "mysql"], default="sqlite", help="Database type (sqlite or mysql)")
    parser.add_argument("--user", default="root", help="MySQL user (for MySQL only)")
    parser.add_argument("--password", default="", help="MySQL password (for MySQL only)")
    parser.add_argument("--host", default="localhost", help="MySQL host (for MySQL only)")
    parser.add_argument("--port", default="3306", help="MySQL port (for MySQL only)")
    parser.add_argument("--name", default="candidate_db", help="Database name")
    
    args = parser.parse_args()
    
    print("AI Resume Analyzer - Database Setup")
    print("===================================")
    
    if args.type == "sqlite":
        print(f"Setting up SQLite database: {args.name}.db")
        success = setup_sqlite_database(f"{args.name}.db")
    else:
        print(f"Setting up MySQL database: {args.name}")
        success = setup_mysql_database(args.user, args.password, args.host, args.port, args.name)
    
    if success:
        print("\nDatabase setup completed successfully!")
        print("\nYou can now run the application:")
        print("1. Start the API: uvicorn api:app --reload")
        print("2. Start the UI: streamlit run main.py")
    else:
        print("\nDatabase setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()