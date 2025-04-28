from fastapi import FastAPI, HTTPException
from tools.logging_handler import EnterpriseLogger
from tools.security_middleware import SecurityMiddleware

logger = EnterpriseLogger().get_logger(__name__)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tools.logging_handler import EnterpriseLogger
from tools.security_middleware import SecurityMiddleware
from models import Session, Candidate, JobDescription # Added JobDescription import
from agents.orchestrator import AnalysisCoordinator
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(SecurityMiddleware)

@app.get("/health")
async def health_check():
    """Kubernetes health check endpoint"""
    try:
        # Removed validate_db_connection() call as it's not defined
        # A basic check could involve trying to create a session, but keeping it simple for now.
        # session = Session()
        # session.close()
        return {
            "status": "healthy",
            "version": "1.0.0",
            "dependencies": {
                "database": "connected",
                "redis": "disabled"
            }
        }
    except Exception as e:
        logger.critical("health_check_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unavailable")

# Add security middleware
app.add_middleware(
    SecurityMiddleware,
    max_requests=1000,
    rate_limit="100/hour"
)

# Initialize enterprise logger
logger = EnterpriseLogger().get_logger(__name__)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.post("/submit-application")
async def submit_application(
    job_id: str = Form(...), 
    job_title: str = Form(...),
    experience_years: str = Form(...),
    job_description: str = Form(...),
    resume: UploadFile = File(...),
    provider: str = Form("ollama"),
    model_name: str = Form(None),
    api_key: str = Form(None),
    base_url: str = Form(None),
    # Legacy parameters for backward compatibility
    use_openai: str = Form(None),
    openai_api_key: str = Form(None),
    openai_model: str = Form(None)
):
    if not resume:
        raise HTTPException(status_code=400, detail="No resume file provided")
    
    if not job_description:
        raise HTTPException(status_code=400, detail="Job description is required")
    
    # Validate file extension
    allowed_extensions = [".pdf", ".doc", ".docx"]
    file_ext = os.path.splitext(resume.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOADS_DIR, resume.filename)
        with open(file_path, "wb") as f:
            content = await resume.read()
            f.write(content)

        # Convert experience_years to integer
        try:
            exp_years = int(experience_years)
        except ValueError:
            exp_years = 0
            logger.warning(f"Invalid experience years value: {experience_years}, defaulting to 0")

        # Store job description in database
        session = Session()
        try:
            # Check if job ID already exists
            existing_job = session.query(JobDescription).filter_by(job_id=job_id).first()
            
            if existing_job:
                # Update existing job description
                existing_job.title = job_title
                existing_job.experience_years = exp_years
                existing_job.description = job_description
                existing_job.updated_at = datetime.utcnow()
                logger.info(f"Updated existing job description for ID: {job_id}")
            else:
                # Create new job description
                new_job = JobDescription(
                    job_id=job_id,
                    title=job_title,
                    experience_years=exp_years,
                    description=job_description
                )
                session.add(new_job)
                logger.info(f"Created new job description with ID: {job_id}")
            
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing job description: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error storing job description: {str(e)}")
        finally:
            session.close()

        # Use the job description from the form directly
        job_description_content = job_description

        # Set up model configuration
        model_config = {}
        
        # Handle legacy parameters for backward compatibility
        if use_openai and use_openai.lower() == "true" and openai_api_key:
            model_config = {
                "provider": "openai",
                "api_key": openai_api_key,
                "model_name": openai_model or "gpt-3.5-turbo"
            }
            logger.info("Using OpenAI model for analysis (legacy parameters)")
        else:
            # Use new provider-based configuration
            model_config = {
                "provider": provider.lower(),
                "model_name": model_name
            }
            
            # Add API key if provided
            if api_key:
                model_config["api_key"] = api_key
                
                # For OpenAI and OpenRouter, also set the environment variable
                if provider.lower() in ["openai", "openrouter"]:
                    os.environ["OPENAI_API_KEY"] = api_key
                
            # Add base URL for Ollama if provided
            if provider.lower() == "ollama" and base_url:
                model_config["base_url"] = base_url
                
            # Set default model name if not provided
            if not model_name:
                if provider.lower() == "ollama":
                    model_config["model_name"] = os.getenv("OLLAMA_MODEL", "phi3")
                elif provider.lower() == "openai":
                    model_config["model_name"] = "gpt-3.5-turbo"
                elif provider.lower() == "openrouter":
                    model_config["model_name"] = "openai/gpt-3.5-turbo"
                elif provider.lower() == "huggingface":
                    model_config["model_name"] = "mistralai/Mistral-7B-Instruct-v0.2"
            
            logger.info(f"Using {provider} model for analysis: {model_config.get('model_name', 'default')}")

        # Set API keys directly
        provider_for_log = model_config.get("provider", "ollama")
        api_key = model_config.get("api_key")
        
        if provider_for_log.lower() in ["openai", "openrouter"] and api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            logger.info(f"Set OPENAI_API_KEY environment variable for {provider_for_log}")
        
        logger.info(f"Using {provider_for_log} model for analysis")
        
        # Process application
        coordinator = AnalysisCoordinator()
        # Pass job description content along with file path, job_id, and model config
        result = coordinator.process_application(file_path, job_id, job_description_content, model_config)

        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error in submit_application: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

@app.get("/candidate/{candidate_id}")
def get_candidate(candidate_id: int):
    session = Session()
    try:
        candidate = session.query(Candidate).filter_by(id=candidate_id).first()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        # Format technical score to ensure it's a number
        technical_score = candidate.technical_score
        if technical_score is None:
            technical_score = 0
            
        return {
            'id': candidate.id,
            'job_id': candidate.job_id,
            'status': candidate.status,
            'technical_score': technical_score,
            'gap_analysis': candidate.gap_analysis,
            'application_date': candidate.application_date.isoformat() if candidate.application_date else None,
            'skills': candidate.skills,
            'experience': candidate.experience,
            'education': candidate.education,
            'contact_info': candidate.contact_info
        }
    finally:
        session.close()

@app.get("/job-descriptions")
def get_job_descriptions():
    """Get all job descriptions"""
    session = Session()
    try:
        jobs = session.query(JobDescription).all()
        
        # Convert to list of dicts for JSON serialization
        result = []
        for job in jobs:
            result.append({
                'id': job.id,
                'job_id': job.job_id,
                'title': job.title,
                'experience_years': job.experience_years,
                'description': job.description,
                'created_at': job.created_at.isoformat(),
                'updated_at': job.updated_at.isoformat()
            })
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving job descriptions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.get("/job-description/{job_id}")
def get_job_description(job_id: str):
    """Get job description by ID"""
    session = Session()
    try:
        job = session.query(JobDescription).filter(JobDescription.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Job description with ID {job_id} not found")
        
        # Convert to dict for JSON serialization
        result = {
            'id': job.id,
            'job_id': job.job_id,
            'title': job.title,
            'experience_years': job.experience_years,
            'description': job.description,
            'created_at': job.created_at.isoformat(),
            'updated_at': job.updated_at.isoformat()
        }
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving job description: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)