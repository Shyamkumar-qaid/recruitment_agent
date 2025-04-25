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

# Input fields
col1, col2 = st.columns(2)
with col1:
    job_id = st.text_input("Job ID", help="Enter the job reference number")
with col2:
    resume_file = st.file_uploader(
        "Upload Resume",
        type=['pdf', 'doc', 'docx'],
        help="Supported formats: PDF, DOC, DOCX"
    )

# Submit button
if st.button("Analyze Resume", type="primary", disabled=not (job_id and resume_file)):
    if resume_file and job_id:
        try:
            with st.spinner("Analyzing resume..."):
                # Create form data
                files = {"resume": resume_file}
                data = {"job_id": job_id}
                
                # Submit to API
                response = requests.post(
                    "http://localhost:8000/submit-application",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Success message
                    st.success("Resume analysis completed!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Technical Score",
                            f"{result.get('technical_score', 0)}%"
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
        except Exception as e:
            st.error(f"Error processing resume: {str(e)}")

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