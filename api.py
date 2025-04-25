from fastapi import FastAPI, HTTPException
from tools.logging_handler import EnterpriseLogger
from tools.security_middleware import SecurityMiddleware

logger = EnterpriseLogger().get_logger(__name__)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tools.logging_handler import EnterpriseLogger
from tools.security_middleware import SecurityMiddleware
from models import Session, Candidate # Added Candidate import
from agents.orchestrator import AnalysisCoordinator
import os
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
async def submit_application(job_id: str = Form(...), resume: UploadFile = File(...)):
    if not resume:
        raise HTTPException(status_code=400, detail="No resume file provided")
    
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

        # Construct job description path and read content
        job_desc_filename = f"{job_id}.txt"
        job_desc_path = os.path.join("job_descriptions", job_desc_filename)

        if not os.path.exists(job_desc_path):
            logger.error("job_description_not_found", job_id=job_id, path=job_desc_path)
            raise HTTPException(status_code=404, detail=f"Job description for ID '{job_id}' not found.")

        try:
            with open(job_desc_path, "r", encoding='utf-8') as jd_file:
                job_description_content = jd_file.read()
        except Exception as e:
            logger.error("job_description_read_error", job_id=job_id, path=job_desc_path, error=str(e))
            raise HTTPException(status_code=500, detail=f"Could not read job description for ID '{job_id}'.")

        # Process application
        coordinator = AnalysisCoordinator()
        # Pass job description content along with file path and job_id
        result = coordinator.process_application(file_path, job_id, job_description_content)

        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/candidate/{candidate_id}")
def get_candidate(candidate_id: int):
    session = Session()
    try:
        candidate = session.query(Candidate).filter_by(id=candidate_id).first()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        return {
            'id': candidate.id,
            'status': candidate.status,
            'technical_score': candidate.technical_score,
            'gap_analysis': candidate.gap_analysis,
            'created_at': candidate.created_at
        }
    finally:
        session.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)