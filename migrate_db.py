import os
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Float, ForeignKey, MetaData, Table
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

def get_database_connection():
    """Create a database connection using environment variables with fallbacks."""
    # Database credentials
    db_user = os.getenv('DB_USER', 'root')
    db_password = os.getenv('DB_PASSWORD', 'root')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '3306')
    db_name = os.getenv('DB_NAME', 'candidate_db')
    db_type = os.getenv('DB_TYPE', 'mysql').lower()
    
    # Encode password to handle special characters
    encoded_password = quote_plus(db_password)
    
    # Construct connection string based on database type
    if db_type == 'mysql':
        connector = 'mysql+mysqlconnector'
        connection_args = f'{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}'
    elif db_type == 'postgresql':
        connector = 'postgresql+psycopg2'
        connection_args = f'{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}'
    elif db_type == 'sqlite':
        # For SQLite, we just need the path
        connector = 'sqlite'
        connection_args = f'///{db_name}.db'
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    database_url = f'{connector}://{connection_args}'
    
    # Create engine
    engine = create_engine(database_url)
    
    return engine

def migrate_database():
    """Create or update database tables."""
    try:
        # Get database connection
        engine = get_database_connection()
        
        # Create metadata object
        metadata = MetaData()
        
        # Define job_descriptions table
        job_descriptions = Table(
            'job_descriptions',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('job_id', String(50), nullable=False, unique=True),
            Column('title', String(100), nullable=False),
            Column('experience_years', Integer, nullable=False),
            Column('description', String(10000), nullable=False),
            Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
            Column('updated_at', DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
        )
        
        # Create tables
        metadata.create_all(engine)
        
        print("Database migration completed successfully.")
        
    except Exception as e:
        print(f"Error during database migration: {str(e)}")

if __name__ == "__main__":
    migrate_database()