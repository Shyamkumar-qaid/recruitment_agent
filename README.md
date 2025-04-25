# AI Resume Analyzer

An intelligent resume analysis system that uses AI to process resumes, evaluate technical skills, perform gap analysis, and provide detailed candidate assessments.

## Features

- Resume parsing and information extraction
- Technical skill evaluation
- Gap analysis against job requirements
- Vector embeddings for semantic search
- Modern Streamlit UI
- RESTful API with FastAPI

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Set up Ollama:**
   - Install Ollama from [https://ollama.com/](https://ollama.com/).
   - Pull a model: `ollama pull phi3` (or another model of your choice)
   - Ensure the Ollama service is running (usually at `http://localhost:11434`).

4. Create a `.env` file with the following variables:

### Troubleshooting Dependency Conflicts

If you encounter dependency conflicts during installation, you can use the provided tools to resolve them:

#### Windows Users
Run the `update_environment.bat` script:
```
update_environment.bat
```

#### Manual Resolution
1. Update pip to the latest version:
```bash
pip install --upgrade pip
```

2. Install dependencies with specific versions:
```bash
pip install pyasn1>=0.6.1,<0.7.0
pip install -r requirements.txt
```

3. If conflicts persist, run the dependency resolution script:
```bash
python update_dependencies.py
```

4. As a last resort, create a fresh virtual environment:
```bash
python -m venv fresh_venv
.\fresh_venv\Scripts\activate
pip install -r requirements.txt
```

### LangChain Import Updates

This project uses LangChain v0.1.0+ which has a new package structure. If you encounter import errors related to LangChain, you can use the provided script to automatically update imports:

```bash
# Install the LangChain CLI
pip install langchain-cli

# Run the import update script
python update_langchain_imports.py
```

The script will scan all Python files in the project and update imports to the new package structure. Common updates include:

- `from langchain.embeddings` → `from langchain_community.embeddings`
- `from langchain.document_loaders` → `from langchain_community.document_loaders`
- `from langchain.text_splitter` → `from langchain_text_splitter`
- `from langchain.prompts` → `from langchain_core.prompts`
- `from langchain.output_parsers` → `from langchain_core.output_parsers`
```env
# Security Configuration
ENCRYPTION_KEY=your_32_byte_encryption_key
SECURITY_ALGORITHM=AES-256-GCM
TOKEN_EXPIRATION=3600

# Monitoring
SENTRY_DSN=your_sentry_dsn

# Ollama Configuration (Ensure Ollama is running)
# Example: OLLAMA_BASE_URL=http://localhost:11434
# Specify the model to use, e.g., llama3
# OLLAMA_MODEL=llama3

# Database Security
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_NAME=candidate_db
DB_SSL_CA=/path/to/ca.pem
DB_SSL_CERT=/path/to/client-cert.pem
DB_SSL_KEY=/path/to/client-key.pem

# Secrets Management
VAULT_ADDR=https://vault.example.com
VAULT_ROLE_ID=your_vault_role
VAULT_SECRET_ID=your_vault_secret
```

4. Create MySQL database:
```sql
CREATE DATABASE candidate_db;
```

## Running the Application

1. Start the FastAPI backend:
```bash
uvicorn api:app --reload
```

2. Start the Streamlit frontend:
```bash
streamlit run main.py
```

3. Access the application:
- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

## Project Structure

```
├── main.py                    # Streamlit UI
├── api.py                     # FastAPI endpoints
├── models.py                  # SQLAlchemy models
├── .gitignore                 # Git ignore file
├── requirements.txt           # Project dependencies
├── DOCUMENTATION.md           # Detailed documentation
├── update_dependencies.py     # Dependency resolution script
├── update_langchain_imports.py # LangChain import updater
├── update_environment.bat     # Environment setup script for Windows
├── agents/
│   ├── orchestrator.py        # Analysis coordinator
│   ├── resume_agent.py        # Resume processing
│   ├── gap_agent.py           # Gap analysis
│   └── tech_eval_agent.py     # Technical evaluation
├── tools/
│   ├── document_processing.py # Document handling
│   ├── llm_tools.py           # LLM utilities
│   ├── security_middleware.py # API security features
│   └── logging_handler.py     # Structured logging
├── uploads/                   # Temporary file storage
├── logs/                      # Application logs
└── job_descriptions/          # Sample job descriptions
```

## Technologies Used

- FastAPI
- Streamlit
- SQLAlchemy
- LangChain
- OpenAI
- Pinecone
- MySQL

## Version Control

This project uses Git for version control. A `.gitignore` file is included to prevent unnecessary files from being tracked:

- **Excluded files**: Python bytecode, virtual environments, IDE files, logs, uploads, and sensitive configuration
- **Included directories**: Empty directories needed for the application structure are preserved with `.gitkeep` files

### Initial Setup for Version Control

```bash
# Initialize a new Git repository
git init

# Add all files (respecting .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit"

# Add a remote repository (replace with your repository URL)
git remote add origin https://github.com/yourusername/ai-resume-analyzer.git

# Push to the remote repository
git push -u origin main
```

## License

QAID Software Private Limited