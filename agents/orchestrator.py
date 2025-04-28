from datetime import datetime
from typing import Dict, Any
from crewai import Agent, Crew
from .resume_agent import ResumeAgent
from .gap_agent import GapAgent
from .tech_eval_agent import TechEvalAgent
from models import Session, Candidate, AuditLog
import os
from dotenv import load_dotenv

# Import the centralized LLM service
from services.llm_service import create_llm_service

load_dotenv()

class AnalysisCoordinator:
    def __init__(self):
        self.resume_agent = None
        self.gap_agent = None
        self.tech_agent = None
        
        # Default model configuration
        self.model_config = {
            "provider": "ollama",
            "model_name": os.getenv("OLLAMA_MODEL", "phi3"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        }
        
        # Initialize LLM service with default configuration
        self.llm_service = create_llm_service(self.model_config)
        
        # Initialize orchestrator agent with default model
        self.agent = Agent(
            role='Recruitment Process Orchestrator',
            goal='Coordinate the complete candidate evaluation process',
            backstory='Expert in managing and coordinating recruitment workflows',
            verbose=True,
            allow_delegation=True,
            llm=f"ollama/{self.model_config['model_name']}"
        )
    
    def _initialize_agents(self, model_config=None):
        """Initialize agents with the appropriate model configuration"""
        if model_config:
            self.model_config.update(model_config)
            # Reinitialize LLM service with updated configuration
            self.llm_service = create_llm_service(self.model_config)
        
        # Set API keys directly
        provider = self.model_config.get("provider", "ollama").lower()
        api_key = self.model_config.get("api_key")
        
        if provider in ["openai", "openrouter"] and api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print(f"Set OPENAI_API_KEY environment variable for {provider} in AnalysisCoordinator._initialize_agents")
        
        # Initialize agents with the current model configuration
        self.resume_agent = ResumeAgent(model_config=self.model_config)
        self.gap_agent = GapAgent(model_config=self.model_config)
        self.tech_agent = TechEvalAgent(model_config=self.model_config)
        
        # Get provider and model name for CrewAI agent
        provider = self.model_config.get("provider", "ollama").lower()
        model_name = self.model_config.get("model_name", os.getenv("OLLAMA_MODEL", "phi3"))
        
        # Map provider to CrewAI format
        provider_map = {
            "ollama": "ollama",
            "openai": "openai",
            "openrouter": "openai",  # OpenRouter uses OpenAI-compatible API
            "huggingface": "huggingface"
        }
        
        crewai_provider = provider_map.get(provider, "ollama")
        
        # Set API keys directly
        if provider in ["openai", "openrouter"] and "api_key" in self.model_config:
            os.environ["OPENAI_API_KEY"] = self.model_config["api_key"]
            print(f"Set OPENAI_API_KEY environment variable for {provider} in AnalysisCoordinator.__init__")
            
        # Update orchestrator agent with the appropriate provider/model
        print(f"Initializing AnalysisCoordinator with {crewai_provider}/{model_name}")
        
        self.agent = Agent(
            role='Recruitment Process Orchestrator',
            goal='Coordinate the complete candidate evaluation process',
            backstory='Expert in managing and coordinating recruitment workflows',
            verbose=True,
            allow_delegation=True,
            llm=f"{crewai_provider}/{model_name}"
        )
        print(f"AnalysisCoordinator initialized with {crewai_provider} model: {model_name}")
    
    def log_audit(self, session: Session, candidate_id: int, action: str, details: Dict[str, Any]) -> None:
        """Record an action in the audit trail"""
        try:
            audit = AuditLog(
                candidate_id=candidate_id,
                action=action,
                details=details,
                timestamp=datetime.utcnow()
            )
            session.add(audit)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error logging audit: {str(e)}")
    
    def process_application(self, resume_path: str, job_id: str, job_description_content: str, model_config=None) -> Dict[str, Any]:
        """Main entry point for processing a new application"""
        # Import our API key manager utility
        from utils.api_key_manager import set_api_keys_from_config
        
        # Set API keys directly instead of using context manager
        set_api_keys_from_config(model_config)
        print(f"Processing application with API keys from config")
            
        # Initialize agents with the provided model configuration
        self._initialize_agents(model_config)
        
        session = Session()
        # Initialize progress tracking
        progress_steps = [
            {"name": "Resume Processing", "status": "pending", "description": "Extracting information from resume"},
            {"name": "Gap Analysis", "status": "pending", "description": "Analyzing gaps between qualifications and requirements"},
            {"name": "Technical Evaluation", "status": "pending", "description": "Evaluating technical skills"},
            {"name": "Final Decision", "status": "pending", "description": "Making final recommendation"}
        ]
        
        try:
            # Create new candidate record
            candidate = Candidate(
                job_id=job_id,
                status='Processing',
                application_date=datetime.utcnow()
            )
            session.add(candidate)
            session.commit()
            
            # Log initial request
            model_type = "OpenAI" if self.model_config.get("use_openai", False) else "Local LLM"
            self.log_audit(
                session=session,
                candidate_id=candidate.id,
                action='application_received',
                details={
                    'job_id': job_id, 
                    'resume_file': resume_path,
                    'model_type': model_type
                }
            )
            
            # Process resume
            progress_steps[0]["status"] = "in_progress"
            resume_result = self.resume_agent.process(resume_path, {'job_id': job_id})
            if resume_result.get('status') == 'error':
                progress_steps[0]["status"] = "error"
                self.log_audit(
                    session=session,
                    candidate_id=candidate.id,
                    action='error',
                    details={'stage': 'resume_processing', 'error': resume_result['error']}
                )
                raise Exception(resume_result['error'])
            progress_steps[0]["status"] = "completed"
            
            # Perform gap analysis
            # Prepare data for gap analysis with validation
            skills = resume_result.get('skills', [])
            if skills is None or not isinstance(skills, list):
                skills = []
                
            experience = resume_result.get('experience', [])
            if experience is None or not isinstance(experience, list):
                experience = []
                
            education = resume_result.get('education', [])
            if education is None or not isinstance(education, list):
                education = []
                
            resume_data_for_gap = {
                'skills': skills,
                'experience': experience,
                'education': education
            }
            progress_steps[1]["status"] = "in_progress"
            gap_result = self.gap_agent.analyze(
                resume_data=resume_data_for_gap,
                job_description=job_description_content
            )
            if gap_result.get('status') == 'error':
                progress_steps[1]["status"] = "error"
                self.log_audit(
                    session=session,
                    candidate_id=candidate.id,
                    action='error',
                    details={'stage': 'gap_analysis', 'error': gap_result['error']}
                )
                raise Exception(gap_result['error'])
            progress_steps[1]["status"] = "completed"
            
            # Initial screening decision
            try:
                # Ensure skills is a list before passing to screening
                skills_for_screening = resume_result.get('skills', [])
                if skills_for_screening is None or not isinstance(skills_for_screening, list):
                    skills_for_screening = ["No specific skills identified"]
                
                screening_result = self._evaluate_screening(
                    session=session,
                    candidate_id=candidate.id,
                    skills=skills_for_screening,
                    gap_analysis=gap_result['analysis']
                )
            except Exception as e:
                self.log_audit(
                    session=session,
                    candidate_id=candidate.id,
                    action='error',
                    details={'stage': 'screening_evaluation', 'error': str(e)}
                )
                raise Exception(f"Error during screening evaluation: {str(e)}")
            
            if not screening_result['proceed']:
                candidate.status = 'Rejected'
                session.commit()
                self.log_audit(
                    session=session,
                    candidate_id=candidate.id,
                    action='rejection',
                    details={'reason': screening_result['reason']}
                )
                return {
                    'status': 'success',
                    'candidate_id': candidate.id,
                    'next_step': 'Rejected',
                    'details': screening_result['reason']
                }
            
            # Perform technical evaluation with validated skills
            try:
                skills = resume_result.get('skills', [])
                if skills is None or not isinstance(skills, list):
                    skills = ["No specific skills identified"]
                elif len(skills) == 0:
                    skills = ["No specific skills identified"]
                    
                progress_steps[2]["status"] = "in_progress"
                tech_result = self.tech_agent.evaluate(
                    candidate_id=candidate.id,
                    skills=skills,
                    job_id=job_id,
                    job_description=job_description_content # Added job description
                )
            except Exception as e:
                progress_steps[2]["status"] = "error"
                self.log_audit(
                    session=session,
                    candidate_id=candidate.id,
                    action='error',
                    details={'stage': 'technical_evaluation_preparation', 'error': str(e)}
                )
                raise Exception(f"Error preparing for technical evaluation: {str(e)}")
            if tech_result.get('status') == 'error':
                progress_steps[2]["status"] = "error"
                self.log_audit(
                    session=session,
                    candidate_id=candidate.id,
                    action='error',
                    details={'stage': 'technical_evaluation', 'error': tech_result['error']}
                )
                raise Exception(tech_result['error'])
            progress_steps[2]["status"] = "completed"
            
            # Update candidate record with validation
            try:
                # Ensure skills is a list
                skills_to_save = resume_result.get('skills', [])
                if skills_to_save is None or not isinstance(skills_to_save, list):
                    skills_to_save = ["No specific skills identified"]
                elif len(skills_to_save) == 0:
                    skills_to_save = ["No specific skills identified"]
                
                candidate.skills = skills_to_save
                candidate.experience = resume_result.get('experience', [])
                candidate.education = resume_result.get('education', [])
                candidate.contact_info = resume_result.get('contact_info', {})
                candidate.embeddings_ref = resume_result.get('embeddings_ref', '')
                candidate.gap_analysis = gap_result.get('analysis', {})
                candidate.technical_score = tech_result.get('score', 0)
                candidate.technical_feedback = tech_result.get('feedback', {})
            except Exception as e:
                self.log_audit(
                    session=session,
                    candidate_id=candidate.id,
                    action='error',
                    details={'stage': 'candidate_record_update', 'error': str(e)}
                )
                raise Exception(f"Error updating candidate record: {str(e)}")
            
            # Make final decision
            progress_steps[3]["status"] = "in_progress"
            try:
                final_decision = self._make_final_decision(
                    session=session,
                    candidate_id=candidate.id,
                    # resume_score is not directly available from resume_result, relying on gap and tech score
                    gap_analysis=gap_result['analysis'], 
                    technical_score=tech_result['score']
                )
                
                candidate.status = final_decision['next_step']
                session.commit()
                progress_steps[3]["status"] = "completed"
            except Exception as e:
                progress_steps[3]["status"] = "error"
                raise e
            
            return {
                'status': 'success',
                'candidate_id': candidate.id,
                'next_step': final_decision['next_step'],
                'details': final_decision['details'],
                'technical_score': tech_result.get('score', 0),
                'progress_steps': progress_steps
            }
            
        except Exception as e:
            session.rollback()
            if 'candidate' in locals():
                candidate.status = 'Error'
                session.commit()
                self.log_audit(
                    session=session,
                    candidate_id=candidate.id,
                    action='error',
                    details={'stage': 'process_application', 'error': str(e)}
                )
            return {
                'status': 'error',
                'error': str(e),
                'progress_steps': progress_steps
            }
        finally:
            session.close()
    
    def _evaluate_screening(self, session: Session, candidate_id: int, skills: list, gap_analysis: Dict) -> Dict:
        """Evaluate initial screening criteria"""
        try:
            # Validate inputs
            if skills is None:
                skills = ["No specific skills identified"]
            elif not isinstance(skills, list):
                try:
                    if isinstance(skills, str):
                        skills = [skill.strip() for skill in skills.split(',') if skill.strip()]
                    else:
                        skills = [str(skills)]
                except Exception:
                    skills = ["Error processing skills"]
            
            # Ensure we have at least one skill
            if len(skills) == 0:
                skills = ["No specific skills identified"]
                
            if gap_analysis is None:
                gap_analysis = {}
            elif not isinstance(gap_analysis, dict):
                gap_analysis = {}
                
            # Implement screening logic here
            # This is a placeholder implementation
            proceed = True
            reason = 'Meets initial criteria'
            
            self.log_audit(
                session=session,
                candidate_id=candidate_id,
                action='screening_decision',
                details={'proceed': proceed, 'reason': reason}
            )
            
            return {'proceed': proceed, 'reason': reason}
        except Exception as e:
            self.log_audit(
                session=session,
                candidate_id=candidate_id,
                action='error',
                details={'stage': 'screening_evaluation', 'error': str(e)}
            )
            return {'proceed': False, 'reason': f'Error in screening: {str(e)}'}
    
    def _make_final_decision(self, session: Session, candidate_id: int, 
                           gap_analysis: Dict, technical_score: float) -> Dict: # Removed resume_score
        """Make final assessment decision"""
        try:
            # Validate inputs
            if gap_analysis is None:
                gap_analysis = {}
            elif not isinstance(gap_analysis, dict):
                gap_analysis = {}
                
            if technical_score is None:
                technical_score = 0
            elif not isinstance(technical_score, (int, float)):
                try:
                    technical_score = float(technical_score)
                except (ValueError, TypeError):
                    technical_score = 0
            
            # Implement decision logic here
            # Example: Use technical score and consider gap analysis (placeholder logic)
            overall_score = technical_score # Using technical score as the primary numeric indicator for now
            
            # Decision logic based on overall score and gap analysis
            if overall_score >= 70 and not self._has_critical_gaps(gap_analysis): # Adjusted threshold and added gap check
                next_step = 'Interview'
                details = 'Candidate meets technical requirements and gap analysis is acceptable.'
            elif overall_score >= 50: # Adjusted threshold
                next_step = 'Further Review'
                details = 'Moderate technical performance, requires further review.'
            else:
                next_step = 'Rejected'
                details = 'Does not meet minimum technical requirements or has critical gaps.'
            
            self.log_audit(
                session=session,
                candidate_id=candidate_id,
                action='final_decision',
                details={'next_step': next_step, 'reason': details}
            )
            
            return {'next_step': next_step, 'details': details}
        except Exception as e:
            self.log_audit(
                session=session,
                candidate_id=candidate_id,
                action='error',
                details={'stage': 'final_decision', 'error': str(e)}
            )
            return {'next_step': 'Error', 'details': str(e)}
            
    def _has_critical_gaps(self, gap_analysis: Dict) -> bool:
        """
        Determine if the candidate has critical gaps that would disqualify them.
        
        Args:
            gap_analysis: Dictionary containing gap analysis results
            
        Returns:
            Boolean indicating whether critical gaps exist
        """
        try:
            # Validate input
            if gap_analysis is None:
                return False
                
            if not isinstance(gap_analysis, dict):
                return False
                
            # Check for missing_skills field
            missing_skills = gap_analysis.get('missing_skills', [])
            if missing_skills is None:
                missing_skills = []
                
            if not isinstance(missing_skills, list):
                if isinstance(missing_skills, str):
                    missing_skills = [missing_skills]
                else:
                    missing_skills = []
            
            # Example logic: If there are more than 5 missing skills, consider it critical
            if len(missing_skills) > 5:
                return True
                
            # Check for experience_gaps field
            experience_gaps = gap_analysis.get('experience_gaps', [])
            if experience_gaps is None:
                experience_gaps = []
                
            if not isinstance(experience_gaps, list):
                if isinstance(experience_gaps, str):
                    experience_gaps = [experience_gaps]
                else:
                    experience_gaps = []
            
            # Example logic: If there are more than 3 experience gaps, consider it critical
            if len(experience_gaps) > 3:
                return True
                
            return False
        except Exception as e:
            # Log the error but don't fail the process
            print(f"Error in _has_critical_gaps: {str(e)}")
            return False