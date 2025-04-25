from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

# Define paths
UPLOADS_DIR = "uploads"
JOB_DESC_DIR = "job_descriptions"
JOB_ID = "JOB001"
RESUME_FILENAME = "test_resume.pdf"
JOB_DESC_FILENAME = f"{JOB_ID}.txt"

# Ensure uploads directory exists
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Define file paths
resume_path = os.path.join(UPLOADS_DIR, RESUME_FILENAME)
job_desc_path = os.path.join(JOB_DESC_DIR, JOB_DESC_FILENAME)

def generate_resume(output_path, job_description_content):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # --- Resume Content ---
    story.append(Paragraph("Jane Doe", styles['h1']))
    story.append(Paragraph("Software Engineer", styles['h2']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Contact:", styles['h3']))
    story.append(Paragraph("Email: jane.doe@email.com | Phone: 123-456-7890 | LinkedIn: linkedin.com/in/janedoe", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Summary:", styles['h3']))
    story.append(Paragraph("Results-oriented Software Engineer with experience in designing, developing, and maintaining web applications. Proficient in Python and related frameworks.", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Skills:", styles['h3']))
    # Extract required skills from job description (simple example)
    required_skills = []
    if "Required Skills:" in job_description_content:
        skills_section = job_description_content.split("Required Skills:")[1].split("\n\n")[0]
        required_skills = [s.strip().lstrip('- ') for s in skills_section.strip().split('\n') if s.strip()]
    
    skills_text = ", ".join(required_skills) if required_skills else "Python, FastAPI, Docker, SQL, Git"
    story.append(Paragraph(f"Technical: {skills_text}", styles['Normal']))
    story.append(Paragraph("Other: Agile Methodologies, Problem Solving, Team Collaboration", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Experience:", styles['h3']))
    story.append(Paragraph("Software Engineer | Tech Solutions Inc. | 2020 - Present", styles['h4']))
    story.append(Paragraph("- Developed and maintained backend services using Python and FastAPI.", styles['Bullet']))
    story.append(Paragraph("- Collaborated with team members to design and implement new features.", styles['Bullet']))
    story.append(Paragraph("- Utilized Docker for containerization and deployment.", styles['Bullet']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Education:", styles['h3']))
    story.append(Paragraph("Bachelor of Science in Computer Science | University of Technology | 2016 - 2020", styles['Normal']))
    story.append(Spacer(1, 12))

    # --- Build PDF ---
    doc.build(story)
    print(f"Test resume generated successfully at: {output_path}")

if __name__ == "__main__":
    job_desc_content = ""
    try:
        with open(job_desc_path, 'r', encoding='utf-8') as f:
            job_desc_content = f.read()
    except FileNotFoundError:
        print(f"Error: Job description file not found at {job_desc_path}")
        # Use default skills if job description is missing
    except Exception as e:
        print(f"Error reading job description: {e}")
        # Use default skills on error

    generate_resume(resume_path, job_desc_content)