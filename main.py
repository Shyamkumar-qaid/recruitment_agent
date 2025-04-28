import streamlit as st
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.upload-box {
    border: 2px dashed #4CAF50;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin: 20px 0;
}
.status-box {
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
}
.success {
    background-color: #E8F5E9;
    border: 1px solid #4CAF50;
}
.error {
    background-color: #FFEBEE;
    border: 1px solid #F44336;
}
.step-progress {
    margin: 20px 0;
}
.step {
    padding: 10px 15px;
    margin: 5px 0;
    border-radius: 5px;
    transition: all 0.3s;
}
.step-pending {
    background-color: #F5F5F5;
    border-left: 3px solid #9E9E9E;
    color: #757575;
}
.step-in-progress {
    background-color: #E3F2FD;
    border-left: 3px solid #2196F3;
    color: #0D47A1;
}
.step-completed {
    background-color: #E8F5E9;
    border-left: 3px solid #4CAF50;
    color: #1B5E20;
}
.step-error {
    background-color: #FFEBEE;
    border-left: 3px solid #F44336;
    color: #B71C1C;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("AI Resume Analyzer")
st.markdown("""
    Upload a resume and get instant AI-powered analysis including:
    - Technical skill assessment
    - Experience evaluation
    - Gap analysis
    - Personalized recommendations
""")

# Model selection and API key
with st.expander("Model Settings", expanded=False):
    st.markdown("### Choose LLM Provider")
    provider_type = st.selectbox(
        "Select LLM provider to use for analysis:",
        ["Ollama (Local)", "OpenAI", "OpenRouter", "Hugging Face"],
        index=0,
        help="Choose which LLM provider to use for analysis. Some providers require API keys."
    )
    
    # Common fields for all providers
    api_key = None
    model_name = None
    base_url = None
    
    if provider_type == "Ollama (Local)":
        st.info("Using local Ollama models. Make sure Ollama is running on your machine.")
        model_name = st.selectbox(
            "Ollama Model",
            ["phi3", "llama3", "mistral", "gemma", "mixtral"],
            index=0,
            help="Select which Ollama model to use"
        )
        base_url = st.text_input(
            "Ollama Base URL (Optional)",
            value="http://localhost:11434",
            help="URL where Ollama is running"
        )
    
    elif provider_type == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use GPT models"
        )
        model_name = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0,
            help="Select which OpenAI model to use"
        )
        st.info("Your API key is not stored and will only be used for this session.")
    
    elif provider_type == "OpenRouter":
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Enter your OpenRouter API key"
        )
        model_name = st.selectbox(
            "OpenRouter Model",
            [
                "openai/gpt-3.5-turbo", 
                "openai/gpt-4-turbo", 
                "anthropic/claude-3-opus", 
                "anthropic/claude-3-sonnet",
                "google/gemini-pro",
                "meta-llama/llama-3-70b-instruct"
            ],
            index=0,
            help="Select which model to use through OpenRouter"
        )
        st.info("OpenRouter provides access to multiple LLM providers through a single API.")
    
    elif provider_type == "Hugging Face":
        api_key = st.text_input(
            "Hugging Face API Token",
            type="password",
            help="Enter your Hugging Face API token"
        )
        model_name = st.text_input(
            "Hugging Face Model ID",
            value="mistralai/Mistral-7B-Instruct-v0.2",
            help="Enter the Hugging Face model ID"
        )
        st.info("Hugging Face provides access to thousands of open-source models.")

# Job Description Section
st.subheader("Job Information")

# Add a tab for creating a new job or selecting an existing one
job_tabs = st.tabs(["Create New Job", "Select Existing Job"])

with job_tabs[0]:
    # Create new job
    job_title = st.text_input("Job Title", help="Enter the job title (e.g., Senior Software Engineer)", key="new_job_title")
    col1, col2 = st.columns(2)
    with col1:
        experience_years = st.number_input("Years of Experience Required", min_value=0, max_value=30, value=3, help="Enter the required years of experience")
    with col2:
        job_id = st.text_input("Job ID (Optional)", help="Leave blank to auto-generate based on title and experience")

with job_tabs[1]:
    # Select existing job
    try:
        response = requests.get("http://localhost:8000/job-descriptions")
        if response.status_code == 200:
            jobs = response.json()
            if jobs:
                job_options = {f"{job['title']} ({job['job_id']})": job for job in jobs}
                selected_job = st.selectbox(
                    "Select Job", 
                    options=list(job_options.keys()),
                    format_func=lambda x: x,
                    help="Select an existing job description"
                )
                
                if selected_job:
                    selected_job_data = job_options[selected_job]
                    job_id = selected_job_data["job_id"]
                    job_title = selected_job_data["title"]
                    experience_years = selected_job_data["experience_years"]
                    job_description = selected_job_data["description"]
                    
                    st.info(f"Selected job: {job_title} with {experience_years} years of experience required")
            else:
                st.info("No existing jobs found. Please create a new job.")
        else:
            st.error("Failed to load existing jobs. Please create a new job.")
    except Exception as e:
        st.error(f"Error loading jobs: {str(e)}")
        st.info("Please create a new job instead.")

# Job Description
if 'job_description' not in locals() or not job_description:
    job_description = st.text_area(
        "Job Description", 
        height=200,
        help="Enter the full job description including requirements, responsibilities, etc."
    )
else:
    job_description = st.text_area(
        "Job Description", 
        value=job_description,
        height=200,
        help="Enter the full job description including requirements, responsibilities, etc."
    )

# Resume Upload Section
st.subheader("Resume Upload")
resume_file = st.file_uploader(
    "Upload Resume",
    type=['pdf', 'doc', 'docx'],
    help="Supported formats: PDF, DOC, DOCX"
)

# Generate Job ID if not provided
if not job_id and job_title:
    # Create a job ID based on title and experience
    import re
    # Convert title to lowercase, replace spaces with underscores, remove special chars
    job_id_base = re.sub(r'[^a-zA-Z0-9\s]', '', job_title.lower()).replace(' ', '_')
    # Add experience years
    job_id = f"{job_id_base}_{experience_years}yr"
    st.info(f"Auto-generated Job ID: {job_id}")

# Submit button
button_disabled = not (job_title and job_description and resume_file)

# Check if API key is required but not provided
if provider_type in ["OpenAI", "OpenRouter", "Hugging Face"] and not api_key:
    button_disabled = True
    st.warning(f"Please enter your {provider_type} API key to proceed.")

if st.button("Analyze Resume", type="primary", disabled=button_disabled):
    if resume_file and job_title and job_description:
        try:
            # Create a placeholder for the stepper UI
            stepper_placeholder = st.empty()
            
            # Initialize progress steps
            progress_steps = [
                {"name": "Resume Processing", "status": "pending", "description": "Extracting information from resume"},
                {"name": "Gap Analysis", "status": "pending", "description": "Analyzing gaps between qualifications and requirements"},
                {"name": "Technical Evaluation", "status": "pending", "description": "Evaluating technical skills"},
                {"name": "Final Decision", "status": "pending", "description": "Making final recommendation"}
            ]
            
            # Function to update the stepper UI
            def update_stepper(steps):
                stepper_html = '<div class="step-progress">'
                for step in steps:
                    status_class = f"step-{step['status']}" if step['status'] != 'pending' else "step-pending"
                    status_icon = "✅" if step['status'] == "completed" else "❌" if step['status'] == "error" else "⏳" if step['status'] == "in_progress" else "⏱️"
                    stepper_html += f'<div class="step {status_class}">{status_icon} <strong>{step["name"]}</strong>: {step["description"]}</div>'
                stepper_html += '</div>'
                stepper_placeholder.markdown(stepper_html, unsafe_allow_html=True)
            
            # Show initial stepper
            update_stepper(progress_steps)
            
            # Create form data
            files = {"resume": resume_file}
            data = {
                "job_id": job_id,
                "job_title": job_title,
                "experience_years": str(experience_years),
                "job_description": job_description
            }
            
            # Map provider type to provider name for API
            provider_map = {
                "Ollama (Local)": "ollama",
                "OpenAI": "openai",
                "OpenRouter": "openrouter",
                "Hugging Face": "huggingface"
            }
            
            # Add model configuration
            provider = provider_map.get(provider_type, "ollama")
            data["provider"] = provider
            data["model_name"] = model_name
            
            # Add API key if provided
            if api_key:
                data["api_key"] = api_key
                
                # For OpenAI and OpenRouter, also set the environment variable
                if provider in ["openai", "openrouter"]:
                    import os
                    os.environ["OPENAI_API_KEY"] = api_key
                
            # Add base URL for Ollama if provided
            if provider == "ollama" and base_url:
                data["base_url"] = base_url
            
            # Submit to API
            with st.spinner(f"Analyzing resume with {provider_type}..."):
                response = requests.post(
                    "http://localhost:8000/submit-application",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Success message
                    st.success("Resume analysis completed!")
                    
                    # Update progress steps if available
                    if 'progress_steps' in result:
                        update_stepper(result['progress_steps'])
                    
                    # Display results
                    results_container = st.container()
                    with results_container:
                        st.subheader("Analysis Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # First try to get technical_score from the initial result
                            tech_score = result.get('technical_score', 0)
                            
                            # If it's not there, we'll fetch it from the detailed results later
                            st.metric(
                                "Technical Score",
                                f"{tech_score}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Application ID",
                                f"#{result.get('candidate_id', '')}"
                            )
                        
                        # Fetch detailed results
                        details = requests.get(
                            f"http://localhost:8000/candidate/{result['candidate_id']}"
                        ).json()
                        
                        # Update technical score if it's available in the detailed results
                        if 'technical_score' in details and details['technical_score'] is not None:
                            # Update the previously displayed metric
                            col1.metric(
                                "Technical Score",
                                f"{details['technical_score']}%"
                            )
                        
                        # Display gap analysis
                        if 'gap_analysis' in details:
                            st.subheader("Gap Analysis")
                            gap_analysis = details['gap_analysis']
                            
                            if isinstance(gap_analysis, str):
                                try:
                                    gap_analysis = json.loads(gap_analysis)
                                except:
                                    pass
                            
                            if isinstance(gap_analysis, dict):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("### Strengths")
                                    for strength in gap_analysis.get('strengths', []):
                                        st.markdown(f"- {strength}")
                                
                                with col2:
                                    st.markdown("### Areas for Improvement")
                                    for area in gap_analysis.get('improvements', []):
                                        st.markdown(f"- {area}")
                                
                                if 'recommendations' in gap_analysis:
                                    st.markdown("### Recommendations")
                                    for rec in gap_analysis['recommendations']:
                                        st.markdown(f"- {rec}")
                else:
                    st.error(f"Error: {response.text}")
                    # Update stepper to show error
                    progress_steps[0]["status"] = "error"
                    update_stepper(progress_steps)
        except Exception as e:
            st.error(f"Error processing resume: {str(e)}")
            # Update stepper to show error
            progress_steps[0]["status"] = "error"
            update_stepper(progress_steps)

# Display sample job IDs for testing
with st.expander("Sample Job IDs for Testing"):
    st.markdown("""
    Use these sample Job IDs for testing:
    - SE001: Senior Software Engineer
    - FE002: Frontend Developer
    - BE003: Backend Engineer
    - FS004: Full Stack Developer
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Powered by AI | Built with Streamlit</div>",
    unsafe_allow_html=True
)