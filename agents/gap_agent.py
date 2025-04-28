from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from crewai import Agent, Task
from tools.llm_tools import LLMTools
import os
import re
import json
import logging
from dotenv import load_dotenv

# Import the centralized LLM service
from services.llm_service import create_llm_service

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class GapAgent:
    """
    Agent responsible for analyzing gaps between candidate qualifications and job requirements.
    
    This agent uses LLM to compare resume data with job descriptions and identify:
    - Missing skills
    - Experience gaps
    - Strengths
    - Recommendations for improvement
    """
    
    def __init__(self, model_config=None):
        self.model_config = model_config or {}
        self.llm_tools = LLMTools(model_config)
        
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
            print(f"Set OPENAI_API_KEY environment variable for {provider} in GapAgent.__init__")
            
        # Initialize CrewAI agent with the appropriate provider/model
        print(f"Initializing GapAgent with {crewai_provider}/{model_name}")
        
        self.agent = Agent(
            role='Gap Analysis Expert',
            goal='Analyze gaps between candidate qualifications and job requirements',
            backstory='Specialized in identifying skill gaps and providing recommendations',
            verbose=True,
            allow_delegation=False,
            llm=f"{crewai_provider}/{model_name}"
        )
        logger.info(f"Gap Analysis Agent initialized with {crewai_provider} model: {model_name}")
    
    def analyze(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """
        Analyze gaps between candidate qualifications and job requirements.
        
        Args:
            resume_data: Dictionary containing candidate skills, experience, and education
            job_description: String containing the job description
            
        Returns:
            Dictionary with analysis results or error information
        """
        # Set API keys directly
        provider = self.model_config.get("provider", "ollama").lower()
        api_key = self.model_config.get("api_key")
        
        if provider in ["openai", "openrouter"] and api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print(f"Set OPENAI_API_KEY environment variable for {provider} in GapAgent.analyze")
        logger.info("Starting gap analysis")
        
        # Validate inputs
        if not self._validate_inputs(resume_data, job_description):
            return {
                'status': 'error',
                'error': 'Invalid input data',
                'details': 'Resume data or job description is missing required fields'
            }
        
        try:
            # Format resume data for better understanding
            skills_str = ", ".join(resume_data.get('skills', [])) if isinstance(resume_data.get('skills'), list) else str(resume_data.get('skills', ''))
            
            # Create a task description with all the necessary information
            task_description = f"""Analyze the resume against the job description. Identify key gaps.

Job Description:
{job_description}

Resume Summary:
- Skills: {skills_str}
- Experience: {resume_data.get('experience')}
- Education: {resume_data.get('education')}

Your task is to:
1. Identify skills mentioned in the job description that are missing from the resume
2. Identify experience requirements that the candidate may not meet
3. Highlight the candidate's strengths relative to the job requirements
4. Provide recommendations for the candidate

Output ONLY a valid JSON object with this structure:
{{
  "gaps_summary": "Concise gap analysis summary here",
  "missing_skills": ["skill1", "skill2"],
  "experience_gaps": ["gap1", "gap2"],
  "strengths": ["strength1", "strength2"],
  "recommendations": ["recommendation1", "recommendation2"]
}}

NO other text, NO explanations, NO markdown.
"""
            
            # Create a task for the agent to execute
            analysis_task = Task(
                description=task_description,
                expected_output="A JSON object with gap analysis results",
                agent=self.agent
            )
            
            # Execute the task
            raw_result = self.agent.execute_task(analysis_task)
            logger.debug(f"LLM raw response received: {raw_result[:100]}...")
            
            # Parse and validate the LLM output
            parsed_result, error = self._parse_llm_output(raw_result)
            
            if error:
                logger.error(f"Error parsing LLM output: {error}")
                return {
                    'status': 'error',
                    'error': error,
                    'raw_output': raw_result[:1000]  # Limit output size
                }
            
            logger.info("Gap analysis completed successfully")
            return {
                'status': 'success',
                'analysis': parsed_result
            }
            
        except Exception as e:
            logger.exception("Unexpected error during gap analysis")
            return {
                'status': 'error',
                'error': f"An unexpected error occurred during gap analysis: {str(e)}",
                'error_type': type(e).__name__
            }
    
    def _validate_inputs(self, resume_data: Dict[str, Any], job_description: str) -> bool:
        """
        Validate that the input data contains the required fields.
        
        Args:
            resume_data: Dictionary containing candidate information
            job_description: String containing the job description
            
        Returns:
            Boolean indicating whether inputs are valid
        """
        if not job_description or not isinstance(job_description, str):
            logger.error("Invalid job description")
            return False
            
        if not resume_data or not isinstance(resume_data, dict):
            logger.error("Invalid resume data format")
            return False
            
        required_fields = ['skills', 'experience', 'education']
        missing_fields = [field for field in required_fields if field not in resume_data]
        
        if missing_fields:
            logger.error(f"Resume data missing required fields: {missing_fields}")
            return False
            
        return True
    
    def _parse_llm_output(self, raw_output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse and validate the LLM output.
        
        Args:
            raw_output: String output from the LLM
            
        Returns:
            Tuple containing:
            - Parsed result dictionary (or None if parsing failed)
            - Error message (or None if parsing succeeded)
        """
        if not raw_output:
            return None, "Empty response from LLM"
        
        try:
            # Use the LLM service's JSON parser
            parsed_result = self.llm_service.parse_json_output(raw_output)
            
            if not parsed_result:
                return None, "Failed to parse LLM output as JSON"
            
            # Validate the structure
            required_fields = ['gaps_summary', 'gap_summary']  # Accept either field name
            found_fields = [field for field in required_fields if field in parsed_result]
            
            if not found_fields:
                return None, f"LLM output missing required summary field"
            
            # Normalize field names if needed
            if 'gap_summary' in parsed_result and 'gaps_summary' not in parsed_result:
                parsed_result['gaps_summary'] = parsed_result['gap_summary']
                
            if not isinstance(parsed_result.get('gaps_summary'), str):
                return None, "gaps_summary must be a string"
                
            # Ensure all fields are of the expected type
            list_fields = ['missing_skills', 'experience_gaps', 'missing_experience', 'strengths', 'recommendations', 'improvements']
            for field in list_fields:
                if field in parsed_result and not isinstance(parsed_result[field], list):
                    parsed_result[field] = [parsed_result[field]] if parsed_result[field] else []
            
            # Normalize field names for consistency
            if 'missing_experience' in parsed_result and 'experience_gaps' not in parsed_result:
                parsed_result['experience_gaps'] = parsed_result['missing_experience']
                
            if 'improvements' in parsed_result and 'recommendations' not in parsed_result:
                parsed_result['recommendations'] = parsed_result['improvements']
            
            return parsed_result, None
            
        except Exception as e:
            return None, f"Error processing LLM output: {str(e)}"