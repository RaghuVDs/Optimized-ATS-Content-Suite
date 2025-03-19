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
    tone: str  # Add the tone parameter
) -> Optional[str]:
    """Generates an email or cover letter using Google Gemini with a specified tone.

    Args:
        name: The applicant's name.
        email: The applicant's email address.
        job_description: The job description text.
        resume_content: The text content of the resume.
        generation_type: Either "Email" or "Cover Letter".
        google_api_key: The Google Gemini API key.
        tone: The desired tone of the generated text (e.g., "Formal", "Friendly").

    Returns:
        The generated email or cover letter text, or None if an error occurs.
    """
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

    if generation_type == "Email":
        prompt = f"""Compose a compelling and assertive job application email to a hiring manager with a {tone.lower()} tone.
        Utilize the following details to craft a persuasive narrative:

        Your Name: {name}
        Your Email: {email}
        Job Description: {job_description}
        Your Resume: {resume_content}

        Emphasize your most impactful skills and experiences that directly align with the job requirements. Clearly articulate your unique value proposition with confidence and precision. The email should be concise, direct, and designed to capture the hiring manager’s attention immediately. Do not include a subject line.
        """

    elif generation_type == "Cover Letter":
        prompt = f"""Craft a powerful and persuasive cover letter for a job application with a {tone.lower()} tone.
        Utilize the details provided below to construct an impactful narrative:

        Your Name: {name}
        Your Email: {email}
        Job Description: {job_description}
        Your Resume: {resume_content}

        Explicitly state the position you are applying for and detail how your top-tier skills and significant experiences, as outlined in your resume, meet and exceed the job requirements. Clearly convey your passion for the role and the company, and demonstrate a confident understanding of how you will add immediate value. If no specific name is provided in the job description, address the letter to "Hiring Manager".
        """

    else:
        return "Invalid generation type selected."

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating {generation_type}: {e}"