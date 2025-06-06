from crewai import Agent
from tools.document_processing import DocumentProcessor
from tools.llm_tools import LLMTools
import os
from dotenv import load_dotenv

# Import the centralized LLM service
from services.llm_service import create_llm_service

load_dotenv()

class ResumeAgent:
    def __init__(self, model_config=None):
        self.doc_processor = DocumentProcessor()
        self.llm_tools = LLMTools(model_config)
        self.model_config = model_config or {}
        
        # Create LLM service from model config
        self.llm_service = create_llm_service(model_config)
        
        # Get provider and model name for CrewAI agent
        provider = self.model_config.get("provider", "ollama").lower()
        
        # Map provider to CrewAI format
        provider_map = {
            "ollama": "ollama",
            "openai": "openai",
            "openrouter": "openai",  # OpenRouter uses OpenAI-compatible API
            "huggingface": "huggingface"
        }
        
        crewai_provider = provider_map.get(provider, "ollama")
        model_name = self.model_config.get("model_name", os.getenv("OLLAMA_MODEL", "phi3"))
        
        # Set API keys directly
        if provider in ["openai", "openrouter"] and "api_key" in self.model_config:
            os.environ["OPENAI_API_KEY"] = self.model_config["api_key"]
            print(f"Set OPENAI_API_KEY environment variable for {provider} in ResumeAgent.__init__")
            
        # Initialize CrewAI agent with the appropriate provider/model
        print(f"Initializing ResumeAgent with {crewai_provider}/{model_name}")
        
        self.agent = Agent(
            role='Resume Processing Expert',
            goal='Extract and analyze resume information',
            backstory='Specialized in parsing and understanding resumes',
            verbose=True,
            allow_delegation=False,
            llm=f"{crewai_provider}/{model_name}"
        )
        print(f"ResumeAgent initialized with {crewai_provider} model: {model_name}")
    
    def process(self, file_path, job_context):
        try:
            # Process document and create embeddings
            doc_result = self.doc_processor.process_resume(file_path)
            if doc_result['status'] == 'error':
                return doc_result
            
            # Extract text from the processed document
            similar_docs = self.doc_processor.search_similar_documents(
                query="Extract all relevant information from this resume",
                top_k=doc_result['num_chunks']
            )
            
            # Combine all text chunks
            full_text = ' '.join([doc['metadata']['text'] for doc in similar_docs['matches']])
            
            # Extract structured information
            entities_result = self.llm_tools.extract_entities(full_text)
            if entities_result.get('status') != 'success':
                # Propagate the error status and details
                return entities_result
            
            # Extract data from the 'data' key upon success
            entities_data = entities_result.get('data', {})
            
            # Ensure skills is always a list, even if empty or None
            try:
                skills = entities_data.get('skills', [])
                if skills is None:
                    skills = []
                elif not isinstance(skills, list):
                    # Convert to list if it's not already
                    if isinstance(skills, str):
                        skills = [skill.strip() for skill in skills.split(',') if skill.strip()]
                    else:
                        skills = [str(skills)]
                # Make sure we have at least one skill if the list is empty
                if len(skills) == 0:
                    skills = ["No specific skills identified"]
            except Exception as e:
                # Fallback to a safe default if any error occurs
                skills = ["Error processing skills"]
            
            # Ensure experience is always a list
            experience = entities_data.get('experience', [])
            if experience is None:
                experience = []
            elif not isinstance(experience, list):
                experience = [experience] if experience else []
            
            # Ensure education is always a list
            education = entities_data.get('education', [])
            if education is None:
                education = []
            elif not isinstance(education, list):
                education = [education] if education else []
            
            # Ensure contact_info is always a dict
            contact_info = entities_data.get('contact_info', {})
            if contact_info is None or not isinstance(contact_info, dict):
                contact_info = {}
            
            # Final validation before returning
            if not isinstance(skills, list):
                skills = ["Error: skills not properly formatted"]
            
            return {
                'status': 'success',
                'skills': skills,
                'experience': experience,
                'education': education,
                'contact_info': contact_info,
                'embeddings_ref': ','.join(doc_result['vector_ids'])
            }
            
        except Exception as e:
            import logging
            logging.exception(f"Error in ResumeAgent.process: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

