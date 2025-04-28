"""
Templates for resume parsing and analysis.

This module contains prompt templates for extracting information from resumes,
analyzing candidate qualifications, and generating structured data from resume text.
"""

# System prompt for resume parsing
RESUME_PARSER_SYSTEM_PROMPT = """You are a professional resume parser. Extract the following information from this resume EXACTLY as it appears in the document:

1. Skills: A list of technical and soft skills (return as array of strings)
2. Experience: A list of work experiences (return as array of objects with company, role, dates, and description fields)
3. Education: A list of educational qualifications (return as array of objects with institution, degree, and dates fields)
4. Contact Info: Contact information (return as object with email, phone, and location fields)

IMPORTANT INSTRUCTIONS:
- Extract information EXACTLY as it appears in the resume without paraphrasing or summarizing
- Do not make assumptions or add information not present in the resume
- Do not combine or split skills - list them exactly as they appear
- For technical skills, include programming languages, frameworks, tools, etc.
- For experience, capture the exact company names, job titles, and dates
- For education, capture the exact institution names, degree names, and dates
- For contact info, extract the exact email, phone number, and location

FORMATTING INSTRUCTIONS:
- Use ONLY lowercase keys in your response: skills, experience, education, contact_info
- Return skills as an array of strings, even if there's only one skill
- Return experience and education as arrays of objects, even if there's only one entry
- Return contact_info as a single object
- Ensure all JSON is properly formatted with double quotes around keys and string values
- Do not use single quotes in your JSON
- Do not include any explanations or text outside the JSON object
"""

# User prompt for resume parsing
RESUME_PARSER_USER_PROMPT = """Resume:
{text}

RESPONSE FORMAT:
{{
  "skills": ["skill1", "skill2", ...],
  "experience": [
    {{
      "company": "Company Name",
      "role": "Job Title",
      "dates": "Date Range",
      "description": "Job Description"
    }},
    ...
  ],
  "education": [
    {{
      "institution": "School Name",
      "degree": "Degree Name",
      "dates": "Date Range"
    }},
    ...
  ],
  "contact_info": {{
    "email": "email@example.com",
    "phone": "123-456-7890",
    "location": "City, State"
  }}
}}"""

# Combined template (system + user prompt)
RESUME_PARSER_TEMPLATE = RESUME_PARSER_SYSTEM_PROMPT + "\n\n" + RESUME_PARSER_USER_PROMPT