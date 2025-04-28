"""
Templates for technical evaluation of candidate skills.

This module contains prompt templates for evaluating a candidate's technical skills
based on their resume and the job requirements.
"""

# System prompt for technical evaluation
TECH_EVAL_SYSTEM_PROMPT = """You are an expert technical evaluator for job candidates.
Your task is to assess the candidate's technical skills and provide a score and detailed feedback.
"""

# User prompt for technical evaluation
TECH_EVAL_USER_PROMPT = """Evaluate the technical skills of this candidate for the specified job.

Candidate Skills: {skills}
Job Description: {job_description}
Job Title: {job_title}

IMPORTANT FORMATTING INSTRUCTIONS:
- Return a valid JSON object with the following structure
- Use double quotes for all strings and keys
- Ensure all arrays are properly formatted, even if empty
- Score should be an integer between 0 and 100

Output format:
{
  "score": 85,
  "summary": "Concise summary of the candidate's technical qualifications",
  "strengths": ["technical strength 1", "technical strength 2"],
  "weaknesses": ["technical weakness 1", "technical weakness 2"],
  "skill_assessment": [
    {
      "skill": "skill name",
      "relevance": "high/medium/low",
      "proficiency": "expert/proficient/beginner/missing"
    }
  ],
  "recommendations": ["recommendation 1", "recommendation 2"]
}

NO other text, NO explanations, NO markdown."""

# Combined template (system + user prompt)
TECH_EVAL_TEMPLATE = TECH_EVAL_SYSTEM_PROMPT + "\n\n" + TECH_EVAL_USER_PROMPT