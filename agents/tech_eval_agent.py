from crewai import Agent
from langchain_community.llms import Ollama
from tools.llm_tools import LLMTools
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import re
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta

# It's generally better to call load_dotenv() at the module level
load_dotenv()

class TechEvalAgent:
    def __init__(self):
        self.llm_tools = LLMTools()
        self.agent = Agent(
            role='Technical Evaluation Expert',
            goal='Evaluate technical skills and provide detailed assessment',
            backstory='Expert in technical skill assessment and scoring',
            verbose=True,
            allow_delegation=False,
            # Ensure Ollama server is running and the model is pulled (e.g. ollama run llama3)
            llm=Ollama(model="phi3") # Replace with your desired Ollama model
        )

    def evaluate(self, candidate_id: int, skills: List[str], job_id: int, job_description: str) -> Dict[str, Any]:
        """Comprehensive technical evaluation including coding, system design, and behavioral assessment, considering the job description."""
        try:
            # Determine evaluation strategy based on job role, skills, and job description
            strategy = self._determine_evaluation_strategy(job_id, skills, job_description)
            evaluation_results = []

            # Execute each type of evaluation based on strategy
            if strategy.get('requires_coding', False): # Using .get() for safer access
                coding_result = self._evaluate_coding(candidate_id, strategy.get('coding_challenge', {}))
                evaluation_results.append(coding_result)

            if strategy.get('requires_system_design', False): # Using .get() for safer access
                design_result = self._evaluate_system_design(candidate_id, strategy.get('design_prompt', {}))
                evaluation_results.append(design_result)

            if strategy.get('requires_behavioral', False): # Using .get() for safer access
                behavioral_result = self._evaluate_behavioral(candidate_id, strategy.get('behavioral_questions', []))
                evaluation_results.append(behavioral_result)

            # Calculate overall technical score
            overall_score = self._calculate_overall_score(evaluation_results)

            # Generate feedback summary
            feedback_summary = self._generate_feedback_summary(evaluation_results)

            return {
                'status': 'success',
                'score': overall_score,
                'feedback': {
                    'evaluations': evaluation_results,
                    'summary': feedback_summary # Use the generated summary
                }
            }

        except Exception as e:
            # Consider logging the error here for debugging
            import logging
            logging.exception(f"Error during evaluation for job {job_id}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _determine_evaluation_strategy(self, job_id: int, skills: List[str], job_description: str) -> Dict[str, Any]:
        """
        Determine appropriate evaluation methods based on job role, skills, and job description.
        
        Args:
            job_id: The job identifier
            skills: List of candidate skills
            job_description: The job description text
            
        Returns:
            Dictionary containing the evaluation strategy
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Determining evaluation strategy for job ID {job_id}")
        
        # Define the expected schema for validation
        strategy_schema = {
            "requires_coding": bool,
            "requires_system_design": bool,
            "requires_behavioral": bool,
            "coding_challenge": dict,
            "design_prompt": dict,
            "behavioral_questions": list
        }
        
        # Create a structured prompt with clear instructions
        prompt = f"""Analyze the job description, job role ID {job_id}, and candidate skills {skills} to determine the technical evaluation strategy.
        
Job Description:
{job_description}

Decide if coding assessment, system design assessment, and behavioral assessment are needed based on the role, skills, and job description.
If coding is needed, suggest a relevant type of coding challenge (e.g., algorithms, data structures, API interaction) based on the job requirements.
If system design is needed, suggest a relevant prompt (e.g., design a URL shortener, design a basic social media feed) based on the job requirements.
If behavioral is needed, suggest 2-3 key behavioral questions relevant to the technical role and job description (e.g., related to teamwork, problem-solving, handling technical debt).

Provide the response strictly in the following JSON format:
{{
    "requires_coding": boolean,
    "requires_system_design": boolean,
    "requires_behavioral": boolean,
    "coding_challenge": {{ "type": "description" }},
    "design_prompt": {{ "prompt": "description" }},
    "behavioral_questions": ["question1", "question2"]
}}

Ensure the JSON is valid. Fill in default empty values if a section is not required (e.g., {{}} for coding_challenge if requires_coding is false, [] for behavioral_questions if requires_behavioral is false).

IMPORTANT: Your response must be valid JSON. Do not include any explanations or text outside the JSON object.
"""

        try:
            # Get raw response from LLM
            raw_result = self.agent.llm(prompt)
            logger.debug(f"Raw LLM response: {raw_result[:100]}...")
            
            # Parse and validate the JSON response
            parsed_result = self._parse_json_with_fallbacks(raw_result)
            
            if not parsed_result:
                logger.warning("Could not parse LLM output as JSON, using default strategy")
                return self._get_default_strategy()
            
            # Validate the structure against our schema
            validation_errors = self._validate_against_schema(parsed_result, strategy_schema)
            
            if validation_errors:
                logger.warning(f"LLM output validation errors: {validation_errors}")
                # Fix the issues rather than failing
                parsed_result = self._fix_strategy_structure(parsed_result)
            
            logger.info("Successfully determined evaluation strategy")
            return parsed_result

        except Exception as e:
            logger.exception(f"Error determining evaluation strategy: {str(e)}")
            return self._get_default_strategy()
    
    def _parse_json_with_fallbacks(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from text with multiple fallback strategies.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        if not text:
            return None
            
        # Strategy 1: Try to parse the entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Strategy 2: Look for JSON-like structure with regex
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
            
        # Strategy 3: Look for JSON with triple backticks (common in LLM outputs)
        try:
            code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_blocks:
                for block in code_blocks:
                    try:
                        return json.loads(block)
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
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
            
        return None
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, type]) -> List[str]:
        """
        Validate data against a simple schema.
        
        Args:
            data: Dictionary to validate
            schema: Dictionary mapping field names to expected types
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for missing required fields
        for field, expected_type in schema.items():
            if field not in data:
                errors.append(f"Missing required field: {field}")
                continue
                
            # Check type
            if not isinstance(data[field], expected_type):
                errors.append(f"Field {field} has wrong type. Expected {expected_type.__name__}, got {type(data[field]).__name__}")
        
        return errors
    
    def _fix_strategy_structure(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix common issues in the strategy structure.
        
        Args:
            strategy: The strategy dictionary with potential issues
            
        Returns:
            Fixed strategy dictionary
        """
        fixed_strategy = strategy.copy()
        
        # Ensure boolean fields are actually booleans
        for field in ['requires_coding', 'requires_system_design', 'requires_behavioral']:
            if field in fixed_strategy:
                # Convert string representations to actual booleans
                if isinstance(fixed_strategy[field], str):
                    fixed_strategy[field] = fixed_strategy[field].lower() in ['true', 'yes', '1']
            else:
                # Default to False if missing
                fixed_strategy[field] = False
        
        # Ensure required objects exist based on boolean flags
        if fixed_strategy.get('requires_coding', False):
            if 'coding_challenge' not in fixed_strategy or not isinstance(fixed_strategy['coding_challenge'], dict):
                fixed_strategy['coding_challenge'] = {"type": "general coding assessment"}
        else:
            fixed_strategy['coding_challenge'] = {}
            
        if fixed_strategy.get('requires_system_design', False):
            if 'design_prompt' not in fixed_strategy or not isinstance(fixed_strategy['design_prompt'], dict):
                fixed_strategy['design_prompt'] = {"prompt": "design a simple system based on the job requirements"}
        else:
            fixed_strategy['design_prompt'] = {}
            
        if fixed_strategy.get('requires_behavioral', False):
            if 'behavioral_questions' not in fixed_strategy or not isinstance(fixed_strategy['behavioral_questions'], list):
                fixed_strategy['behavioral_questions'] = ["Describe a challenging project you worked on."]
        else:
            fixed_strategy['behavioral_questions'] = []
            
        return fixed_strategy
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """
        Get a default evaluation strategy when LLM fails.
        
        Returns:
            Default strategy dictionary
        """
        return {
            'requires_coding': True,
            'requires_system_design': True,
            'requires_behavioral': True,
            'coding_challenge': {
                'type': 'general programming task',
                'description': 'Implement a function to solve a common programming problem'
            },
            'design_prompt': {
                'prompt': 'Design a scalable system for a common use case',
                'focus_areas': ['scalability', 'reliability', 'maintainability']
            },
            'behavioral_questions': [
                'Describe a challenging technical problem you solved recently.',
                'How do you approach working in a team?',
                'How do you handle tight deadlines?'
            ]
        }


    def _evaluate_coding(self, candidate_id: int, challenge_config: Dict) -> Dict[str, Any]:
        """Evaluate coding skills through practical challenges."""
        # NOTE: This currently just asks the LLM to *generate* an assessment,
        # not *run* code or evaluate actual candidate code submission.
        # The prompt needs candidate's actual code/solution to evaluate.
        # This needs significant expansion for real-world use (e.g., integrating with a coding platform API or parsing code submissions).
        prompt = f"""Based on the coding challenge configuration {challenge_config}, evaluate the hypothetical coding performance for candidate {candidate_id}.
        Assume a hypothetical submission (or provide one if needed).
        Assess factors like correctness, efficiency, code style, and test case coverage.
        Provide a score (0-100) and structured feedback.
        Format as JSON:
        {{
            "score": integer,
            "feedback": {{
                "strengths": ["strength1", "strength2"],
                "improvements": ["improvement1"],
                "recommendations": ["recommendation1"]
            }}
        }}"""

        raw_result = self.agent.llm(prompt)
        parsed_result = None
        try:
            json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_result = json.loads(json_str)
            else:
                parsed_result = json.loads(raw_result)

            if not isinstance(parsed_result, dict) or 'score' not in parsed_result or 'feedback' not in parsed_result:
                raise ValueError("LLM output JSON structure is incorrect for coding assessment.")

            return {
                'type': 'coding',
                'score': parsed_result.get('score', 0),
                'feedback': parsed_result.get('feedback', {})
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing coding assessment JSON: {e}. Raw: '{raw_result[:500]}'...")
            return {'type': 'coding', 'score': 0, 'feedback': {'error': f'Failed to parse assessment: {e}'}}

    def _evaluate_system_design(self, candidate_id: int, design_config: Dict) -> Dict[str, Any]:
        """Evaluate system design capabilities."""
        # NOTE: Similar to coding, this needs the candidate's actual design solution.
        prompt = f"""Based on the system design prompt {design_config}, evaluate the hypothetical system design solution provided by candidate {candidate_id}.
        Assume a hypothetical solution (or provide one).
        Assess architecture choices, scalability, reliability, trade-offs, and clarity of explanation.
        Provide a score (0-100) and structured feedback.
        Format as JSON:
        {{
            "score": integer,
            "feedback": {{
                "strengths": ["strength1", "strength2"],
                "improvements": ["improvement1"],
                "recommendations": ["recommendation1"]
            }}
        }}"""

        raw_result = self.agent.llm(prompt)
        parsed_result = None
        try:
            json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_result = json.loads(json_str)
            else:
                parsed_result = json.loads(raw_result)

            if not isinstance(parsed_result, dict) or 'score' not in parsed_result or 'feedback' not in parsed_result:
                raise ValueError("LLM output JSON structure is incorrect for system design assessment.")

            return {
                'type': 'system_design',
                'score': parsed_result.get('score', 0),
                'feedback': parsed_result.get('feedback', {})
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing system design assessment JSON: {e}. Raw: '{raw_result[:500]}'...")
            return {'type': 'system_design', 'score': 0, 'feedback': {'error': f'Failed to parse assessment: {e}'}}

    def _evaluate_behavioral(self, candidate_id: int, questions: List[str]) -> Dict[str, Any]:
        """Evaluate behavioral competencies."""
         # NOTE: Needs candidate's actual responses to the questions.
        prompt = f"""Given the behavioral questions {questions}, evaluate the hypothetical responses from candidate {candidate_id}.
        Assume hypothetical responses demonstrating relevant competencies.
        Assess communication, problem-solving approach, teamwork/collaboration indicators based on the assumed responses.
        Provide a score (0-100) and structured feedback.
        Format as JSON:
        {{
            "score": integer,
            "feedback": {{
                "strengths": ["strength1", "strength2"],
                "improvements": ["improvement1"],
                "recommendations": ["recommendation1"]
            }}
        }}"""

        raw_result = self.agent.llm(prompt)
        parsed_result = None
        try:
            json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_result = json.loads(json_str)
            else:
                parsed_result = json.loads(raw_result)

            if not isinstance(parsed_result, dict) or 'score' not in parsed_result or 'feedback' not in parsed_result:
                raise ValueError("LLM output JSON structure is incorrect for behavioral assessment.")

            return {
                'type': 'behavioral',
                'score': parsed_result.get('score', 0),
                'feedback': parsed_result.get('feedback', {})
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing behavioral assessment JSON: {e}. Raw: '{raw_result[:500]}'...")
            return {'type': 'behavioral', 'score': 0, 'feedback': {'error': f'Failed to parse assessment: {e}'}}

    def _calculate_overall_score(self, evaluation_results: List[Dict]) -> float:
        """Calculate weighted overall technical score."""
        weights = {
            'coding': 0.4,
            'system_design': 0.4,
            'behavioral': 0.2
        }

        total_score = 0.0 # Use float for intermediate calculations
        total_weight = 0.0

        for result in evaluation_results:
            result_type = result.get('type')
            score = result.get('score', 0)
            if result_type in weights:
                 # Ensure score is a number
                try:
                    numeric_score = float(score)
                    weight = weights[result_type]
                    total_score += numeric_score * weight
                    total_weight += weight
                except (ValueError, TypeError):
                    print(f"Warning: Invalid score type '{score}' for type '{result_type}'. Skipping.")


        if total_weight == 0:
            return 0.0

        # Normalize the score based on the weights used
        return round(total_score / total_weight, 2)

    def _generate_feedback_summary(self, evaluation_results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive feedback summary."""
        # Flatten the feedback from individual evaluations
        all_strengths = []
        all_improvements = []
        all_recommendations = []

        for result in evaluation_results:
            feedback = result.get('feedback', {})
            if isinstance(feedback, dict): # Check if feedback is a dict
                all_strengths.extend(feedback.get('strengths', []))
                all_improvements.extend(feedback.get('improvements', []))
                all_recommendations.extend(feedback.get('recommendations', []))

        return {
            # Use list() to ensure they are lists even if empty
            'strengths': list(all_strengths),
            'areas_for_improvement': list(all_improvements),
            'recommendations': list(all_recommendations)
        }
    # --- The misplaced 'except' block from the original code has been removed ---