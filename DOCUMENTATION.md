# AI Resume Analyzer - Technical Documentation

## System Architecture

The AI Resume Analyzer is built with a modular architecture that separates concerns into distinct components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Streamlit UI   │────▶│   FastAPI API   │────▶│  Agent System   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                        │
                                ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │     MySQL DB    │     │  Vector Store   │
                        │                 │     │   (Pinecone)    │
                        └─────────────────┘     └─────────────────┘
```

### Components

1. **Streamlit UI** (`main.py`)
   - Provides a user-friendly interface for uploading resumes and selecting job IDs
   - Displays analysis results including technical scores, gap analysis, and recommendations

2. **FastAPI Backend** (`api.py`)
   - RESTful API for handling resume uploads and processing
   - Endpoints for submitting applications and retrieving candidate information
   - Implements security middleware for rate limiting and request validation

3. **Agent System** (`agents/`)
   - Orchestrator (`agents/orchestrator.py`): Coordinates the analysis workflow
   - Resume Agent (`agents/resume_agent.py`): Extracts structured information from resumes
   - Gap Agent (`agents/gap_agent.py`): Analyzes differences between candidate qualifications and job requirements
   - Tech Evaluation Agent (`agents/tech_eval_agent.py`): Assesses technical skills and provides scoring

4. **Data Processing Tools** (`tools/`)
   - Document Processing (`tools/document_processing.py`): Handles resume parsing and vector embeddings
   - LLM Tools (`tools/llm_tools.py`): Utilities for working with language models
   - Security Middleware (`tools/security_middleware.py`): Implements API security features
   - Logging Handler (`tools/logging_handler.py`): Enterprise-grade structured logging

5. **Database** (`models.py`)
   - SQLAlchemy models for storing candidate information and audit logs
   - Supports multiple database backends (MySQL, PostgreSQL, SQLite)

## Data Flow

### Resume Submission Flow

1. User uploads a resume through the Streamlit UI
2. UI sends the resume file and job ID to the FastAPI backend
3. API validates the input and saves the resume temporarily
4. Orchestrator coordinates the analysis process:
   - Resume Agent extracts structured information (skills, experience, education)
   - Gap Agent compares resume to job description
   - Tech Evaluation Agent assesses technical skills
5. Results are stored in the database
6. API returns the analysis results to the UI
7. UI displays the results to the user

### Vector Database Integration

1. Document Processing tool splits the resume into chunks
2. Each chunk is embedded using HuggingFace models
3. Embeddings are stored in Pinecone for semantic search
4. Vector IDs are stored in the database for reference

## API Reference

### Endpoints

#### `POST /submit-application`

Submit a new job application for analysis.

**Request:**
- Form data:
  - `job_id` (string): The job identifier
  - `resume` (file): The resume file (PDF, DOC, or DOCX)

**Response:**
```json
{
  "status": "success",
  "candidate_id": 123,
  "next_step": "Interview",
  "details": "Candidate meets technical requirements and gap analysis is acceptable."
}
```

#### `GET /candidate/{candidate_id}`

Retrieve detailed information about a candidate.

**Parameters:**
- `candidate_id` (integer): The candidate identifier

**Response:**
```json
{
  "id": 123,
  "status": "Interview",
  "technical_score": 85,
  "gap_analysis": {
    "gaps_summary": "Strong technical skills but lacks experience in cloud deployment",
    "missing_skills": ["AWS", "CI/CD"],
    "strengths": ["Python", "FastAPI", "SQL"],
    "recommendations": ["Gain experience with cloud platforms", "Learn CI/CD tools"]
  },
  "created_at": "2023-04-15T14:30:00Z"
}
```

#### `GET /health`

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "dependencies": {
    "database": "connected",
    "redis": "disabled"
  }
}
```

## Database Schema

### Candidates Table

| Column            | Type      | Description                                |
|-------------------|-----------|--------------------------------------------|
| id                | Integer   | Primary key                                |
| job_id            | String    | Job reference ID                           |
| status            | String    | Application status                         |
| application_date  | DateTime  | When the application was submitted         |
| skills            | JSON      | Extracted skills                           |
| experience        | JSON      | Work experience                            |
| education         | JSON      | Educational background                     |
| contact_info      | JSON      | Contact information                        |
| embeddings_ref    | String    | Reference to vector embeddings             |
| gap_analysis      | JSON      | Gap analysis results                       |
| technical_score   | Float     | Technical evaluation score                 |
| technical_feedback| JSON      | Detailed technical feedback                |

### Audit Logs Table

| Column       | Type      | Description                                |
|--------------|-----------|--------------------------------------------|
| id           | Integer   | Primary key                                |
| candidate_id | Integer   | Foreign key to candidates table            |
| action       | String    | Action performed                           |
| details      | JSON      | Additional details about the action        |
| timestamp    | DateTime  | When the action occurred                   |

## LLM Integration

The system uses Ollama for local LLM inference. The default model is `phi3`, but this can be configured through environment variables.

### LLM Prompting Strategies

1. **Resume Information Extraction**
   - Structured prompts to extract skills, experience, education, and contact information
   - Output parsing with validation and error handling

2. **Gap Analysis**
   - Comparison of resume data with job description
   - Identification of missing skills and experience
   - Strengths and recommendations

3. **Technical Evaluation**
   - Determination of appropriate evaluation methods based on job role
   - Simulated assessment of coding, system design, and behavioral competencies
   - Weighted scoring based on job requirements

## Configuration

The application uses environment variables for configuration. Create a `.env` file with the following variables:

```env
# Database Configuration
DB_TYPE=mysql                # mysql, postgresql, or sqlite
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=candidate_db
DB_SSL_CA=/path/to/ca.pem    # Optional
DB_SSL_CERT=/path/to/cert.pem # Optional
DB_SSL_KEY=/path/to/key.pem   # Optional

# LLM Configuration
OLLAMA_MODEL=phi3            # Model to use with Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Vector Database (Pinecone)
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=us-east1-gcp
PINECONE_INDEX_NAME=candidate-embeddings

# Logging and Monitoring
LOG_LEVEL=INFO               # DEBUG, INFO, WARNING, ERROR, CRITICAL
SENTRY_DSN=your_sentry_dsn   # Optional
```

## Error Handling

The application implements comprehensive error handling at multiple levels:

1. **Input Validation**
   - File format validation
   - Required field validation
   - Data type validation

2. **LLM Output Parsing**
   - Multiple parsing strategies for handling different LLM output formats
   - Fallback mechanisms when parsing fails
   - Schema validation for expected output structure

3. **Database Errors**
   - Connection error handling
   - Transaction management with rollback on error
   - Fallback to SQLite for development environments

4. **API Error Responses**
   - Structured error responses with status codes and details
   - Rate limiting and security error handling
   - Comprehensive logging of errors

## Logging

The application uses structured logging with the following features:

1. **Context-aware Logging**
   - Request context tracking
   - User and action tracking
   - Environment and version information

2. **Log Formats**
   - JSON format for production
   - Console format for development
   - Configurable log levels

3. **Integration with Monitoring**
   - Sentry integration for error tracking
   - Support for distributed tracing

## Security Considerations

1. **API Security**
   - Rate limiting to prevent abuse
   - Input validation to prevent injection attacks
   - Content-Type validation

2. **Database Security**
   - Parameterized queries to prevent SQL injection
   - SSL/TLS support for database connections
   - Password encoding for special characters

3. **File Handling**
   - Temporary file storage with cleanup
   - File type validation
   - Size limits for uploads

## Development Guidelines

### Adding New Features

1. **New Agent Types**
   - Create a new file in the `agents/` directory
   - Implement the agent interface with `process` or `analyze` methods
   - Update the orchestrator to include the new agent

2. **New API Endpoints**
   - Add the endpoint to `api.py`
   - Implement appropriate validation and error handling
   - Update the documentation

3. **Database Schema Changes**
   - Update the models in `models.py`
   - Create a migration script if using Alembic
   - Update the documentation

### Testing

1. **Unit Tests**
   - Test individual components in isolation
   - Mock external dependencies
   - Focus on edge cases and error handling

2. **Integration Tests**
   - Test the interaction between components
   - Use test databases and mock LLMs
   - Verify end-to-end workflows

3. **Performance Testing**
   - Test with large resumes and complex job descriptions
   - Measure response times and resource usage
   - Identify bottlenecks and optimize

## Deployment

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Ollama:
   - Install Ollama from [https://ollama.com/](https://ollama.com/)
   - Pull a model: `ollama pull phi3`
   - Start the Ollama service

4. Create a `.env` file with your configuration

5. Start the services:
```bash
# Start the API
uvicorn api:app --reload --port 8000

# Start the UI (in a separate terminal)
streamlit run main.py
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t resume-analyzer .
```

2. Run the container:
```bash
docker run -p 8000:8000 -p 8501:8501 --env-file .env resume-analyzer
```

### Production Deployment

For production deployment, consider:

1. Using a reverse proxy (Nginx, Traefik) for SSL termination and load balancing
2. Setting up a production-grade database with proper backup and replication
3. Implementing proper authentication and authorization
4. Setting up monitoring and alerting
5. Using container orchestration (Kubernetes, Docker Swarm) for scaling

## Troubleshooting

### Common Issues

1. **LLM Connection Errors**
   - Ensure Ollama is running and accessible
   - Check that the correct model is pulled
   - Verify the OLLAMA_BASE_URL environment variable

2. **Database Connection Issues**
   - Check database credentials
   - Verify network connectivity
   - Ensure the database exists and has the correct schema

3. **Vector Database Issues**
   - Verify Pinecone API key and environment
   - Check that the index exists with the correct dimension
   - Monitor rate limits and quotas

4. **Performance Problems**
   - LLM inference can be slow, especially for large documents
   - Consider optimizing chunk size for document processing
   - Monitor memory usage, especially with large files

### Debugging

1. Set `LOG_LEVEL=DEBUG` in your `.env` file
2. Check the application logs for detailed information
3. Use the `/health` endpoint to verify service status
4. For LLM issues, check the raw outputs in the error responses

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.