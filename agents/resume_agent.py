from crewai import Agent
from tools.document_processing import DocumentProcessor
from tools.llm_tools import LLMTools
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()

class ResumeAgent:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.llm_tools = LLMTools()
        # Get model name from environment variable with fallback
        model_name = os.getenv("OLLAMA_MODEL", "phi3")
        
        self.agent = Agent(
            role='Resume Processing Expert',
            goal='Extract and analyze resume information',
            backstory='Specialized in parsing and understanding resumes',
            verbose=True,
            allow_delegation=False,
            # Use the litellm format: provider/model
            llm="ollama/phi3"
        )
    
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
            skills = entities_data.get('skills', [])
            if skills is None:
                skills = []
            elif not isinstance(skills, list):
                # Convert to list if it's not already
                if isinstance(skills, str):
                    skills = [skill.strip() for skill in skills.split(',') if skill.strip()]
                else:
                    skills = [str(skills)]
            
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
            
            return {
                'status': 'success',
                'skills': skills,
                'experience': experience,
                'education': education,
                'contact_info': contact_info,
                'embeddings_ref': ','.join(doc_result['vector_ids'])
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

