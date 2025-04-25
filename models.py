from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import logging
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

Base = declarative_base()

class Candidate(Base):
    __tablename__ = 'candidates'
    id = Column(Integer, primary_key=True)
    job_id = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    application_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    skills = Column(JSON)
    experience = Column(JSON)
    education = Column(JSON)
    contact_info = Column(JSON)
    embeddings_ref = Column(String(255))
    gap_analysis = Column(JSON)
    technical_score = Column(Float)
    technical_feedback = Column(JSON)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey('candidates.id'), nullable=False)
    action = Column(String(100), nullable=False)
    details = Column(JSON)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

def get_database_connection():
    """
    Create a database connection using environment variables with fallbacks.
    Supports multiple database types and SSL configuration.
    """
    # Database credentials
    db_user = os.getenv('DB_USER', 'root')
    db_password = os.getenv('DB_PASSWORD', '')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '3306')
    db_name = os.getenv('DB_NAME', 'candidate_db')
    db_type = os.getenv('DB_TYPE', 'mysql').lower()
    
    # Encode password to handle special characters
    encoded_password = quote_plus(db_password)
    
    # SSL Configuration (if provided)
    ssl_args = {}
    ssl_ca = os.getenv('DB_SSL_CA')
    ssl_cert = os.getenv('DB_SSL_CERT')
    ssl_key = os.getenv('DB_SSL_KEY')
    
    if ssl_ca and ssl_cert and ssl_key:
        ssl_args = {
            'ssl_ca': ssl_ca,
            'ssl_cert': ssl_cert,
            'ssl_key': ssl_key
        }
    
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
        logger.error(f"Unsupported database type: {db_type}")
        raise ValueError(f"Unsupported database type: {db_type}")
    
    database_url = f'{connector}://{connection_args}'
    
    # Log connection attempt (without sensitive info)
    logger.info(f"Connecting to {db_type} database at {db_host}:{db_port}/{db_name}")
    
    # Create engine with SSL if configured
    if ssl_args and db_type != 'sqlite':
        engine = create_engine(database_url, connect_args=ssl_args)
    else:
        engine = create_engine(database_url)
    
    return engine

# Initialize database connection
try:
    engine = get_database_connection()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    logger.info("Database connection established successfully")
except Exception as e:
    logger.error(f"Database connection failed: {str(e)}")
    # Fallback to SQLite for development/testing
    if os.getenv('ENVIRONMENT', 'development') != 'production':
        logger.warning("Falling back to SQLite database for development")
        engine = create_engine('sqlite:///candidate_db.db')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
    else:
        # In production, we want to fail if the database connection fails
        raise