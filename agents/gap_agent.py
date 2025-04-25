from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from crewai import Agent
from tools.llm_tools import LLMTools
from langchain_community.llms import Ollama
import os
import re
import json
import logging
from dotenv import load_dotenv

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
    
    def __init__(self):
        self.llm_tools = LLMTools()
        # Get model name from environment variable with fallback
        model_name = os.getenv("OLLAMA_MODEL", "phi3")
        
        self.agent = Agent(
            role='Gap Analysis Expert',
            goal='Analyze gaps between candidate qualifications and job requirements',
            backstory='Specialized in identifying skill gaps and providing recommendations',
            verbose=True,
            allow_delegation=False,
            llm=Ollama(model=model_name)
        )
        logger.info(f"Gap Analysis Agent initialized with model: {model_name}")
    
    def analyze(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """
        Analyze gaps between candidate qualifications and job requirements.
        
        Args:
            resume_data: Dictionary containing candidate skills, experience, and education
            job_description: String containing the job description
            
        Returns:
            Dictionary with analysis results or error information
        """
        logger.info("Starting gap analysis")
        
        # Validate inputs
        if not self._validate_inputs(resume_data, job_description):
            return {
                'status': 'error',
                'error': 'Invalid input data',
                'details': 'Resume data or job description is missing required fields'
            }
        
        try:
            # Create a structured prompt for the LLM
            prompt = self._create_analysis_prompt(resume_data, job_description)
            
            # Invoke the LLM
            raw_result = self.agent.llm.invoke(prompt)
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
    
    def _create_analysis_prompt(self, resume_data: Dict[str, Any], job_description: str) -> str:
        """
        Create a structured prompt for the LLM to analyze gaps.
        
        Args:
            resume_data: Dictionary containing candidate information
            job_description: String containing the job description
            
        Returns:
            Formatted prompt string
        """
        # Format resume data for better LLM understanding
        skills_str = ", ".join(resume_data.get('skills', [])) if isinstance(resume_data.get('skills'), list) else str(resume_data.get('skills', ''))
        
        # Create a structured prompt with clear instructions
        prompt = f"""Analyze the resume against the job description. Identify key gaps.

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
        return prompt
    
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
            # Try finding JSON within potentially messy output
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_result = json.loads(json_str)
            else:
                # Fallback: try parsing the whole string
                parsed_result = json.loads(raw_output)
            
            # Validate the structure
            required_fields = ['gaps_summary']
            missing_fields = [field for field in required_fields if field not in parsed_result]
            
            if missing_fields:
                return None, f"LLM output missing required fields: {missing_fields}"
                
            if not isinstance(parsed_result.get('gaps_summary'), str):
                return None, "gaps_summary must be a string"
                
            # Ensure all fields are of the expected type
            list_fields = ['missing_skills', 'experience_gaps', 'strengths', 'recommendations']
            for field in list_fields:
                if field in parsed_result and not isinstance(parsed_result[field], list):
                    parsed_result[field] = [parsed_result[field]] if parsed_result[field] else []
            
            return parsed_result, None
            
        except json.JSONDecodeError as e:
            return None, f"Failed to parse LLM output as JSON: {str(e)}"
            
        except Exception as e:
            return None, f"Error processing LLM output: {str(e)}"