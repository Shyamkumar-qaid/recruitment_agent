"""
Templates for gap analysis between candidate qualifications and job requirements.

This module contains prompt templates for analyzing the gaps between a candidate's
qualifications and the requirements of a job position.
"""

# System prompt for gap analysis
GAP_ANALYSIS_SYSTEM_PROMPT = """You are an expert in analyzing gaps between candidate qualifications and job requirements.
Your task is to identify strengths, weaknesses, and areas for improvement based on the candidate's profile and the job description.
"""

# User prompt for gap analysis
GAP_ANALYSIS_USER_PROMPT = """Analyze gaps between candidate profile and job requirements.

Candidate Profile: {profile}
Job Description: {job_desc}

IMPORTANT FORMATTING INSTRUCTIONS:
- Return a valid JSON object with the following structure
- Use double quotes for all strings and keys
- Ensure all arrays are properly formatted, even if empty

Output format:
{
  "gap_summary": "Concise summary of the candidate's fit for the role",
  "missing_skills": ["skill1", "skill2"],
  "missing_experience": ["experience gap 1", "experience gap 2"],
  "strengths": ["strength1", "strength2"],
  "improvements": ["area for improvement 1", "area for improvement 2"],
  "recommendations": ["recommendation 1", "recommendation 2"]
}

NO other text, NO explanations, NO markdown."""

# Combined template (system + user prompt)
GAP_ANALYSIS_TEMPLATE = GAP_ANALYSIS_SYSTEM_PROMPT + "\n\n" + GAP_ANALYSIS_USER_PROMPT