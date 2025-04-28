"""
Enhanced resume parsing module to improve extraction accuracy.
"""

import re
import logging
from typing import Dict, Any, List, Optional
import json

# Import the centralized LLM service
from services.llm_service import LLMService, create_llm_service
from templates.resume_templates import RESUME_PARSER_TEMPLATE

logger = logging.getLogger(__name__)

class ResumeParser:
    """
    Enhanced resume parser that uses multiple techniques to extract information accurately.
    """
    
    def __init__(self, llm_service=None, model_config=None):
        """
        Initialize the resume parser with an optional LLM service.
        
        Args:
            llm_service: Optional LLMService instance
            model_config: Optional model configuration dictionary
        """
        # Use provided LLM service or create a new one
        self.llm_service = llm_service or create_llm_service(model_config)
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse resume text into structured data.
        
        Args:
            text: The resume text to parse
            
        Returns:
            Dictionary containing parsed resume data
        """
        # First, try to extract structured data using the LLM
        llm_result = self._extract_with_llm(text)
        
        # Then, use regex to validate and enhance the extraction
        enhanced_result = self._enhance_with_regex(text, llm_result)
        
        return enhanced_result
    
    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from resume text using LLM.
        
        Args:
            text: The resume text to analyze
            
        Returns:
            Dictionary with extracted entities
        """
        # Truncate text if too long
        max_length = 8000
        if len(text) > max_length:
            logger.warning(f"Resume text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length] + "..."
        
        try:
            # Use the template from the templates module
            result = self.llm_service.run_prompt_template(
                RESUME_PARSER_TEMPLATE,
                {"text": text}
            )
            
            logger.debug(f"Raw LLM response: {result[:100]}...")
            
            # Parse the JSON response using the LLM service's parser
            parsed_data = self.llm_service.parse_json_output(result)
            
            if not parsed_data:
                logger.error("Failed to parse LLM output as JSON")
                return {
                    "skills": [],
                    "experience": [],
                    "education": [],
                    "contact_info": {}
                }
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error extracting entities with LLM: {str(e)}")
            return {
                "skills": [],
                "experience": [],
                "education": [],
                "contact_info": {}
            }
    
    def _enhance_with_regex(self, text: str, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance LLM extraction with regex-based validation and extraction.
        
        Args:
            text: The resume text
            llm_result: The result from LLM extraction
            
        Returns:
            Enhanced extraction result
        """
        enhanced = llm_result.copy()
        
        # Extract email with regex if not found by LLM
        if not enhanced.get('contact_info', {}).get('email'):
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            if email_match:
                if 'contact_info' not in enhanced:
                    enhanced['contact_info'] = {}
                enhanced['contact_info']['email'] = email_match.group(0)
        
        # Extract phone with regex if not found by LLM
        if not enhanced.get('contact_info', {}).get('phone'):
            phone_match = re.search(r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
            if phone_match:
                if 'contact_info' not in enhanced:
                    enhanced['contact_info'] = {}
                enhanced['contact_info']['phone'] = phone_match.group(0)
        
        # Ensure skills is a list
        if 'skills' not in enhanced or not enhanced['skills']:
            enhanced['skills'] = []
        elif not isinstance(enhanced['skills'], list):
            enhanced['skills'] = [enhanced['skills']]
        
        # Ensure experience is a list of dicts
        if 'experience' not in enhanced or not enhanced['experience']:
            enhanced['experience'] = []
        elif not isinstance(enhanced['experience'], list):
            enhanced['experience'] = [enhanced['experience']] if enhanced['experience'] else []
        
        # Ensure education is a list of dicts
        if 'education' not in enhanced or not enhanced['education']:
            enhanced['education'] = []
        elif not isinstance(enhanced['education'], list):
            enhanced['education'] = [enhanced['education']] if enhanced['education'] else []
        
        # Ensure contact_info is a dict
        if 'contact_info' not in enhanced or not enhanced['contact_info']:
            enhanced['contact_info'] = {}
        
        return enhanced