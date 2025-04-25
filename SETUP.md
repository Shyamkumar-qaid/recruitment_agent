# Setup Guide

This guide provides detailed instructions for setting up and running the AI Resume Analyzer project. Follow these steps carefully to ensure a proper development environment.

## Prerequisites

- Python 3.8 or higher
- MySQL Server
- Git (for version control)
- Ollama (ensure it's running locally)
- Pinecone API key and environment

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Virtual Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: If you encounter any issues with Pillow installation, try:
```bash
pip install --no-binary pillow pillow
```

### 4. Setup Ollama

1.  **Install Ollama:** Follow the official instructions for your operating system at [https://ollama.com/](https://ollama.com/).
2.  **Pull a Model:** Download a model like Llama 3.
    ```bash
    ollama pull llama3
    ```
3.  **Ensure Ollama is Running:** Start the Ollama service/application. By default, it runs on `http://localhost:11434`.

### 5. Database Setup

```bash
pip install -r requirements.txt
```

Note: If you encounter any issues with Pillow installation, try:
```bash
pip install --no-binary pillow pillow
```

### 4. Database Setup

1. Install MySQL if not already installed
2. Create a new database:
```sql
CREATE DATABASE candidate_db;
```

### 5. Environment Configuration

1. Create a `.env` file in the project root:
```env
# Environment Type
ENV=development  # [development|staging|production]

# Security Configuration
SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_32_byte_encryption_key

# Ollama Configuration (Ensure Ollama is running)
# Example: OLLAMA_BASE_URL=http://localhost:11434
# Specify the model to use, e.g., llama3
# OLLAMA_MODEL=llama3

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=candidate-embeddings
PINECONE_TIMEOUT=30

# Database Configuration
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=candidate_db
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Monitoring
SENTRY_DSN=your_sentry_dsn

# Rate Limiting
API_RATE_LIMIT=100/hour

# Application Settings
MAX_FILE_SIZE=5242880  # 5MB
ALLOWED_FILE_TYPES=.pdf,.docx,.txt
```

Create `.env.production` for production settings and keep `.env` for development. Never commit actual environment files to version control.

## Running the Application

### Development Mode

1. Start the FastAPI backend:
```bash
uvicorn api:app --reload --port 8000
```

2. Start the Streamlit frontend (in a new terminal):
```bash
streamlit run main.py
```

### Accessing the Application

- Frontend UI: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- API Redoc: http://localhost:8000/redoc

## Development Guidelines

### Project Structure

- `agents/`: Contains AI agent implementations
- `tools/`: Utility functions and tools
- `main.py`: Streamlit frontend
- `api.py`: FastAPI backend
- `models.py`: Database models

### Adding New Features

1. Create new agents in the `agents/` directory
2. Add new utility functions in `tools/`
3. Update API endpoints in `api.py`
4. Add UI components in `main.py`

### Testing

1. Run unit tests:
```bash
python -m pytest tests/
```

2. Test API endpoints using Swagger UI at http://localhost:8000/docs

## Troubleshooting

### Common Issues

1. Database Connection:
- Verify MySQL is running
- Check database credentials in `.env`
- Ensure database exists

2. API Key/Connection Issues:
- Verify Pinecone API key and environment
- Ensure Ollama service is running and accessible

3. Dependencies:
- Try removing `venv` and recreating it
- Update pip: `python -m pip install --upgrade pip`
- Install dependencies one by one if batch install fails

### Getting Help

- Check the project's issue tracker
- Review the documentation
- Contact the development team

## Deployment

Refer to the deployment guide for production setup instructions.