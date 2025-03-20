# llm_handler.py
import google.generativeai as genai
from typing import Optional

def generate_application_text(
    name: str,
    email: str,
    job_description: str,
    resume_content: str,
    generation_type: str,
    google_api_key: str,
    tone: str
) -> Optional[str]:
    """Generates an ATS-friendly email or cover letter using Google Gemini with a specified tone."""
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

    ats_instructions = "Ensure the generated content is optimized for Applicant Tracking Systems (ATS). Use clear and concise language, standard section headings (if applicable), and incorporate relevant keywords from the job description. Avoid complex formatting."

    if generation_type == "Email":
        prompt = f"""Write a concise job application email to a hiring manager with a {tone.lower()} tone. {ats_instructions}
        Use the following information:

        Your Name: {name}
        Your Email: {email}
        Job Description: {job_description}
        Your Resume: {resume_content}

        Focus on highlighting the most relevant skills and experiences from your resume that match the job description. Keep the tone professional and enthusiastic. Do not include a subject line.
        """
    elif generation_type == "Cover Letter":
        prompt = f"""Write a compelling cover letter for a job application with a {tone.lower()} tone. {ats_instructions}
        Use the following information:

        Your Name: {name}
        Your Email: {email}
        Job Description: {job_description}
        Your Resume: {resume_content}

        Clearly state the position you are applying for and highlight how your skills and experience, as detailed in your resume, align with the requirements mentioned in the job description. Express your enthusiasm for the role and the company. Address it to "Hiring Manager" if no specific name is provided in the job description.
        """
    else:
        return "Invalid generation type selected."

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating {generation_type}: {e}"

# Function to generate a tailored resume based on job description
def generate_tailored_resume(default_resume: str, job_description: str, google_api_key: str) -> Optional[str]:
    """Generates a tailored resume based on the default resume and job description."""
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

    prompt = f"""Given the following default resume and job description, generate a tailored resume that highlights the most relevant skills and experiences for the job. Ensure the resume is ATS-friendly.

    Default Resume:
    {default_resume}

    Job Description:
    {job_description}

    Focus on:
    - Updating the summary to align with the job description.
    - Emphasizing relevant skills and achievements in the experience section.
    - Using keywords from the job description where appropriate.
    - Maintaining standard resume formatting.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating tailored resume: {e}"