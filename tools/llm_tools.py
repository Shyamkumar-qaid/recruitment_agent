from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import re
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ResumeEntities(BaseModel):
    """
    Pydantic model for structured resume data.
    
    This model defines the expected structure for extracted resume information,
    including skills, experience, education, and contact details.
    """
    skills: List[str] = Field(..., description="List of technical and soft skills")
    experience: List[dict] = Field(..., description="List of work experiences")
    education: List[dict] = Field(..., description="List of educational qualifications")
    contact_info: dict = Field(..., description="Contact information")

    model_config = {
        "validate_by_name": True,  # Updated from allow_population_by_field_name
        "extra": "ignore"  # Ignore unexpected fields
    }

class LLMTools:
    """
    Utilities for working with Language Models.
    
    This class provides tools for extracting structured information from text,
    parsing LLM outputs, and handling common LLM-related tasks.
    """
    
    def __init__(self):
        """Initialize LLM tools with configured model."""
        # Get model name from environment variable with fallback
        model_name = os.getenv("OLLAMA_MODEL", "phi3")
        self.llm = Ollama(model=model_name)
        self.parser = PydanticOutputParser(pydantic_object=ResumeEntities)
        logger.info(f"LLM Tools initialized with model: {model_name}")
    
    def _normalize_resume_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize keys to match Pydantic model with multiple common variants.
        
        Args:
            data: Dictionary with potentially non-standard keys
            
        Returns:
            Dictionary with normalized keys
        """
        key_mapping = {
            'Skills': 'skills',
            'Skill': 'skills',
            'TechnicalSkills': 'skills',
            'SoftSkills': 'skills',
            'Experience': 'experience',
            'WorkHistory': 'experience',
            'WorkExperience': 'experience',
            'ProfessionalExperience': 'experience',
            'Education': 'education',
            'Academic': 'education',
            'AcademicHistory': 'education',
            'Educational': 'education',
            'Contact_info': 'contact_info',
            'ContactInfo': 'contact_info',
            'Contact': 'contact_info',
            'PersonalInfo': 'contact_info',
            'PersonalInformation': 'contact_info'
        }
        
        # Convert all keys to lowercase for case-insensitive matching
        normalized = {}
        for k, v in data.items():
            # Check for exact match in key_mapping first
            if k in key_mapping:
                normalized[key_mapping[k]] = v
            else:
                # Try lowercase version
                normalized[key_mapping.get(k.lower(), k.lower())] = v
                
        return normalized

    def _handle_validation_error(self, error: ValidationError, raw_data: dict) -> dict:
        """
        Create detailed error report with normalization diagnostics.
        
        Args:
            error: Validation error from Pydantic
            raw_data: The data that failed validation
            
        Returns:
            Dictionary with detailed error information
        """
        error_messages = [f"{err['loc'][0]}: {err['msg']}" for err in error.errors()]
        
        # Log the error for debugging
        logger.warning(f"Validation error: {' | '.join(error_messages)}")
        
        # Provide helpful suggestions based on common issues
        suggestions = []
        for err in error.errors():
            field = err['loc'][0]
            msg = err['msg']
            
            if "missing" in msg.lower():
                suggestions.append(f"Add the '{field}' field")
            elif "not a valid" in msg.lower():
                expected_type = msg.split("not a valid")[1].strip()
                suggestions.append(f"Ensure '{field}' is a {expected_type}")
        
        return {
            'status': 'validation_error',
            'error': " | ".join(error_messages),
            'normalized_data': raw_data,
            'original_data': raw_data,
            'suggestions': suggestions
        }

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract structured entities from resume text with robust error handling.
        
        Args:
            text: The resume text to analyze
            
        Returns:
            Dictionary with extraction results or error information
        """
        # Truncate text if too long to avoid LLM context limits
        max_length = 8000  # Adjust based on your model's context window
        if len(text) > max_length:
            logger.warning(f"Resume text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length] + "..."
        
        # Create a structured prompt with clear instructions
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract the following information from this resume:
1. Skills: A list of technical and soft skills
2. Experience: A list of work experiences with company, role, dates, and descriptions
3. Education: A list of educational qualifications with institution, degree, and dates
4. Contact Info: Contact information including email, phone, and location

Use ONLY lowercase keys in your response: skills, experience, education, contact_info.

Resume:
{text}

{format_instructions}

IMPORTANT: Your response must be valid JSON. Do not include any explanations or text outside the JSON object.""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        result = None
        try:
            # Get raw response from LLM
            result = LLMChain(llm=self.llm, prompt=prompt).run(text=text)
            logger.debug(f"Raw LLM response: {result[:100]}...")
            
            # Parse the JSON response with multiple fallback strategies
            parsed_data, error = self._parse_json_with_fallbacks(result)
            
            if error:
                logger.error(f"JSON parsing error: {error}")
                return {
                    'status': 'json_error',
                    'error': error,
                    'raw_output': result[:1000]  # Limit output size
                }
            
            # Normalize keys to handle variations
            normalized_data = self._normalize_resume_keys(parsed_data)
            
            try:
                # Validate against our Pydantic model
                validated = ResumeEntities(**normalized_data)
                logger.info("Successfully extracted and validated resume entities")
                return {'status': 'success', 'data': validated.dict()}
            except ValidationError as e:
                logger.warning("Validation error during entity extraction")
                return self._handle_validation_error(e, normalized_data)
                
        except Exception as e:
            logger.exception("Unexpected error during entity extraction")
            return {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__,
                'raw_output': result[:1000] if result else None
            }
    
    def _parse_json_with_fallbacks(self, text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse JSON from text with multiple fallback strategies.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Tuple containing:
            - Parsed JSON object (or None if parsing failed)
            - Error message (or None if parsing succeeded)
        """
        if not text:
            return None, "Empty response from LLM"
            
        # Strategy 1: Try to parse the entire text as JSON
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            pass
            
        # Strategy 2: Look for JSON-like structure with regex
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0)), None
        except json.JSONDecodeError:
            pass
            
        # Strategy 3: Look for JSON with triple backticks (common in LLM outputs)
        try:
            code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_blocks:
                for block in code_blocks:
                    try:
                        return json.loads(block), None
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
            
        # Strategy 4: Try to fix common JSON errors and parse again
        try:
            # Replace single quotes with double quotes (common LLM mistake)
            fixed_text = re.sub(r"'([^']*)':\s*", r'"\1": ', text)
            # Fix boolean values
            fixed_text = re.sub(r':\s*True', ': true', fixed_text)
            fixed_text = re.sub(r':\s*False', ': false', fixed_text)
            
            # Try to find and parse JSON in the fixed text
            json_match = re.search(r'\{.*\}', fixed_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0)), None
        except json.JSONDecodeError:
            pass
            
        return None, "Failed to parse LLM output as JSON after multiple attempts"

class GapAgent:
    def __init__(self, llm_tools: LLMTools):
        self.tools = llm_tools
        self.llm = Ollama(model="phi3")

    def _validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize input data with detailed error reporting"""
        try:
            normalized = self.tools._normalize_resume_keys(data)
            validated = ResumeEntities(**normalized)
            return {'is_valid': True, 'data': validated.dict()}
        except ValidationError as e:
            return {
                'is_valid': False,
                'error': self.tools._handle_validation_error(e, normalized),
                'original_data': data
            }

    def analyze(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """Main analysis workflow with enhanced error handling"""
        validation = self._validate_input(resume_data)
        if not validation['is_valid']:
            return {
                'status': 'input_error',
                'error': validation['error'],
                'original_data': validation.get('original_data')
            }

        try:
            prompt = f"""Analyze gaps between candidate profile and job requirements.
                        Candidate Profile: {json.dumps(validation['data'])}
                        Job Description: {job_description}
                        
                        Output format:
                        {{
                            "gap_summary": "string",
                            "missing_skills": ["list"],
                            "missing_experience": ["list"],
                            "strengths": ["list"]
                        }}"""
            
            raw_result = self.llm(prompt)
            return self._parse_output(raw_result)
            
        except Exception as e:
            return {
                'status': 'analysis_error',
                'error': str(e),
                'raw_output': raw_result if 'raw_result' in locals() else None
            }

    def _parse_output(self, raw_result: str) -> Dict[str, Any]:
        """Parse and validate analysis output"""
        try:
            result = json.loads(raw_result)
            required_keys = ['gap_summary', 'missing_skills', 'missing_experience', 'strengths']
            if all(k in result for k in required_keys):
                return {'status': 'success', 'analysis': result}
            return {
                'status': 'output_format_error',
                'error': f"Missing keys: {[k for k in required_keys if k not in result]}",
                'raw_output': raw_result
            }
        except json.JSONDecodeError:
            return {
                'status': 'invalid_json',
                'error': "Analysis output is not valid JSON",
                'raw_output': raw_result[:500]  # Truncate for safety
            }

# Usage Example
if __name__ == "__main__":
    tools = LLMTools()
    analyzer = GapAgent(tools)
    
    # Test with problematic data from the error
    problematic_resume = {
        "Skills": ["Front-end: HTML, CSS, JavaScript"],
        "Experience": [{"company": "Jobcheck", "role": ""}],
        "Education": [{"institution": "Vidya Jyothi", "degree": ""}],
        "Contact_info": {"email": "test@example.com"}
    }
    
    result = analyzer.analyze(problematic_resume, "Need full-stack developer")
    print(json.dumps(result, indent=2))