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

# Import the centralized LLM service
from services.llm_service import LLMService, create_llm_service

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
    
    def __init__(self, model_config=None):
        """
        Initialize LLM tools with configured model.
        
        Args:
            model_config: Dictionary containing model configuration
                - provider: LLM provider name (ollama, openai, openrouter, huggingface)
                - api_key: API key for the provider
                - model_name: Name of the model to use
                - base_url: Base URL for the provider API (optional)
                - temperature: Temperature for text generation (default: 0.2)
                - max_tokens: Maximum tokens to generate (optional)
        """
        # Create LLM service from model config
        self.llm_service = create_llm_service(model_config)
        
        # Store the LLM from the service for compatibility with existing code
        self.llm = self.llm_service.llm
        
        # Store model config for reference
        self.model_config = model_config or {}
        
        # Initialize Pydantic parser
        self.parser = PydanticOutputParser(pydantic_object=ResumeEntities)
    
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
            'PersonalInformation': 'contact_info',
            'PersonalInfo': 'contact_info'
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
        
        # Ensure all required fields exist with proper types
        if 'skills' not in normalized:
            normalized['skills'] = []
        elif not isinstance(normalized['skills'], list):
            # Convert string or other types to list
            if isinstance(normalized['skills'], str):
                normalized['skills'] = [s.strip() for s in normalized['skills'].split(',')]
            else:
                normalized['skills'] = [str(normalized['skills'])]
        
        if 'experience' not in normalized:
            normalized['experience'] = []
        elif not isinstance(normalized['experience'], list):
            # Convert to list if not already
            normalized['experience'] = [normalized['experience']] if normalized['experience'] else []
        
        if 'education' not in normalized:
            normalized['education'] = []
        elif not isinstance(normalized['education'], list):
            # Convert to list if not already
            normalized['education'] = [normalized['education']] if normalized['education'] else []
        
        if 'contact_info' not in normalized:
            normalized['contact_info'] = {}
        elif not isinstance(normalized['contact_info'], dict):
            # Try to convert to dict if possible
            try:
                if isinstance(normalized['contact_info'], str):
                    # Attempt to parse as JSON
                    import json
                    normalized['contact_info'] = json.loads(normalized['contact_info'])
                else:
                    normalized['contact_info'] = {}
            except:
                normalized['contact_info'] = {}
                
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
        # Import the enhanced resume parser
        from tools.resume_parser import ResumeParser
        
        try:
            # Create a resume parser with the current LLM service
            parser = ResumeParser(llm_service=self.llm_service)
            
            # Parse the resume
            parsed_data = parser.parse(text)
            
            if not parsed_data:
                logger.error("Resume parsing failed")
                return {
                    'status': 'error',
                    'error': 'Failed to parse resume content'
                }
            
            # Normalize the parsed data
            normalized_data = self._normalize_resume_keys(parsed_data)
            
            # Validate against our Pydantic model
            try:
                validated = ResumeEntities(**normalized_data)
                logger.info("Successfully extracted and validated resume entities")
                return {'status': 'success', 'data': validated.dict()}
            except ValidationError as e:
                logger.warning("Validation error during entity extraction")
                # Try to fix common validation issues
                try:
                    logger.info("Attempting to fix validation issues")
                    error_fields = [err["loc"][0] for err in e.errors()]
                    
                    # If we're missing required fields, add empty defaults
                    for field in error_fields:
                        if field == "skills" and (field not in normalized_data or not normalized_data[field]):
                            normalized_data[field] = ["No skills specified"]
                        elif field == "experience" and (field not in normalized_data or not normalized_data[field]):
                            normalized_data[field] = [{"company": "Not specified", "role": "Not specified"}]
                        elif field == "education" and (field not in normalized_data or not normalized_data[field]):
                            normalized_data[field] = [{"institution": "Not specified", "degree": "Not specified"}]
                        elif field == "contact_info" and (field not in normalized_data or not normalized_data[field]):
                            normalized_data[field] = {"email": "Not specified"}
                    
                    # Try validation again
                    validated = ResumeEntities(**normalized_data)
                    logger.info("Successfully fixed validation issues")
                    return {'status': 'success', 'data': validated.dict()}
                except Exception as fix_error:
                    logger.error(f"Failed to fix validation issues: {str(fix_error)}")
                    return self._handle_validation_error(e, normalized_data)
            
            # If validation passes, return the normalized data
            return {
                'status': 'success',
                'data': normalized_data
            }
            
        except Exception as e:
            logger.exception(f"Error extracting entities: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
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
        
        # Log the raw text for debugging
        logger.debug(f"Attempting to parse JSON from text: {text[:200]}...")
            
        # Strategy 1: Try to parse the entire text as JSON
        try:
            return json.loads(text), None
        except json.JSONDecodeError as e:
            logger.debug(f"Strategy 1 failed: {str(e)}")
            pass
            
        # Strategy 2: Look for JSON-like structure with regex
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug(f"Found JSON-like structure: {json_str[:100]}...")
                return json.loads(json_str), None
        except json.JSONDecodeError as e:
            logger.debug(f"Strategy 2 failed: {str(e)}")
            pass
            
        # Strategy 3: Look for JSON with triple backticks (common in LLM outputs)
        try:
            code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_blocks:
                logger.debug(f"Found {len(code_blocks)} code blocks")
                for i, block in enumerate(code_blocks):
                    try:
                        logger.debug(f"Trying code block {i+1}: {block[:100]}...")
                        return json.loads(block), None
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse code block {i+1}: {str(e)}")
                        continue
        except Exception as e:
            logger.debug(f"Strategy 3 failed: {str(e)}")
            pass
            
        # Strategy 4: Try to fix common JSON errors and parse again
        try:
            # Replace single quotes with double quotes (common LLM mistake)
            fixed_text = re.sub(r"'([^']*)':\s*", r'"\1": ', text)
            # Fix boolean values
            fixed_text = re.sub(r':\s*True', ': true', fixed_text)
            fixed_text = re.sub(r':\s*False', ': false', fixed_text)
            # Fix trailing commas in arrays and objects
            fixed_text = re.sub(r',\s*}', '}', fixed_text)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            # Fix missing quotes around keys
            fixed_text = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', fixed_text)
            
            logger.debug(f"Fixed text: {fixed_text[:200]}...")
            
            # Try to find and parse JSON in the fixed text
            json_match = re.search(r'\{.*\}', fixed_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug(f"Found JSON in fixed text: {json_str[:100]}...")
                return json.loads(json_str), None
        except json.JSONDecodeError as e:
            logger.debug(f"Strategy 4 failed: {str(e)}")
            pass
            
        # Strategy 5: Last resort - try to construct a minimal valid JSON
        try:
            # Extract key-value pairs using regex
            kv_pairs = re.findall(r'["\']?([a-zA-Z0-9_]+)["\']?\s*:\s*["\']?([^,}\]]+)["\']?', text)
            if kv_pairs:
                logger.debug(f"Found {len(kv_pairs)} key-value pairs")
                constructed_json = {}
                for k, v in kv_pairs:
                    # Clean and normalize the value
                    v = v.strip().strip('"\'')
                    if v.lower() == 'true':
                        constructed_json[k] = True
                    elif v.lower() == 'false':
                        constructed_json[k] = False
                    elif v.isdigit():
                        constructed_json[k] = int(v)
                    else:
                        constructed_json[k] = v
                
                if constructed_json:
                    logger.debug(f"Constructed JSON: {constructed_json}")
                    return constructed_json, None
        except Exception as e:
            logger.debug(f"Strategy 5 failed: {str(e)}")
            pass
            
        return None, "Failed to parse LLM output as JSON after multiple attempts"

class GapAgent:
    def __init__(self, llm_tools: LLMTools = None, model_config=None):
        if llm_tools:
            self.tools = llm_tools
            self.llm_service = llm_tools.llm_service
        else:
            self.tools = LLMTools(model_config)
            self.llm_service = self.tools.llm_service
        
        # Store the LLM from the service for compatibility with existing code
        self.llm = self.llm_service.llm

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
        # Import the gap analysis template
        from templates.gap_analysis_templates import GAP_ANALYSIS_TEMPLATE
        
        validation = self._validate_input(resume_data)
        if not validation['is_valid']:
            return {
                'status': 'input_error',
                'error': validation['error'],
                'original_data': validation.get('original_data')
            }

        try:
            # Use the template from the templates module
            raw_result = self.llm_service.run_prompt_template(
                GAP_ANALYSIS_TEMPLATE,
                {
                    "profile": json.dumps(validation['data']),
                    "job_desc": job_description
                }
            )
            
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
            # Use the LLM service's JSON parser
            result = self.llm_service.parse_json_output(raw_result)
            
            if not result:
                return {
                    'status': 'invalid_json',
                    'error': "Failed to parse LLM output as JSON",
                    'raw_output': raw_result[:500]  # Truncate for safety
                }
            
            # Check for required keys
            required_keys = ['gap_summary', 'missing_skills', 'missing_experience', 'strengths']
            missing_keys = [k for k in required_keys if k not in result]
            
            if missing_keys:
                # Try to fix missing keys with default values
                for key in missing_keys:
                    if key == 'gap_summary':
                        result[key] = "No summary provided"
                    else:
                        result[key] = []
            
            # Ensure all list fields are actually lists
            list_fields = ['missing_skills', 'missing_experience', 'strengths', 'improvements', 'recommendations']
            for field in list_fields:
                if field in result and not isinstance(result[field], list):
                    if result[field] is None or result[field] == "":
                        result[field] = []
                    else:
                        # Convert to list if it's a string or other type
                        result[field] = [result[field]]
            
            return {'status': 'success', 'analysis': result}
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Error parsing output: {str(e)}",
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