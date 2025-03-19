# pdf_handler.py
import PyPDF2
import streamlit as st
from typing import Optional

def read_pdf(uploaded_file) -> Optional[str]:
    """Reads text content from a PDF file.

    Args:
        uploaded_file: An uploaded file object (from Streamlit's file_uploader).

    Returns:
        The extracted text content as a string, or None if an error occurs.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
    
# llm_handler.py
import google.generativeai as genai
from typing import Optional

def generate_application_text(
    name: str,
    email: str,
    job_description: str,
    resume_content: str,
    generation_type: str,
    google_api_key: str
) -> Optional[str]:
    """Generates an email or cover letter using Google Gemini.

    Args:
        name: The applicant's name.
        email: The applicant's email address.
        job_description: The job description text.
        resume_content: The text content of the resume.
        generation_type: Either "Email" or "Cover Letter".
        google_api_key: The Google Gemini API key.

    Returns:
        The generated email or cover letter text, or None if an error occurs.
    """
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

    if generation_type == "Email":
        prompt = f"""Write a concise job application email to a hiring manager.
        Use the following information:

        Your Name: {name}
        Your Email: {email}
        Job Description: {job_description}
        Your Resume: {resume_content}

        Focus on highlighting the most relevant skills and experiences from your resume that match the job description. Keep the tone professional and enthusiastic. Do not include a subject line.
        """
    elif generation_type == "Cover Letter":
        prompt = f"""Write a compelling cover letter for a job application.
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
    

# app.py
import streamlit as st
from typing import Optional
from pdf_handler import read_pdf
from llm_handler import generate_application_text
import os  # Keep os if you plan to use it later

def lab2():
    st.markdown(
        "<h1 style='text-align: center;'>📧📄 Automated Job Application Text Generation</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Enter your details and the job description to generate an email or cover letter.</p>",
        unsafe_allow_html=True,
    )

    # API key handling for Google Gemini
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API key not found in secrets.")
        st.stop()

    # Input fields
    name = st.text_input("Your Name:")
    email = st.text_input("Your Email Address:")
    job_description = st.text_area("Job Description:")

    # Option to upload resume
    uploaded_resume = st.file_uploader("Upload your Resume (PDF only):", type=["pdf"])

    # Load default resume from PDF
    default_resume_content: Optional[str] = None
    try:
        with open("default_resume.pdf", "rb") as default_resume_file:
            default_resume_content = read_pdf(default_resume_file)
        if not default_resume_content:
            st.warning("Could not read the default resume PDF.")
    except FileNotFoundError:
        st.warning("Default resume PDF (default_resume.pdf) not found in the current directory.")
    except Exception as e:
        st.error(f"Error loading default resume: {e}")

    # Determine which resume to use
    resume_content: Optional[str] = default_resume_content
    if uploaded_resume:
        pdf_text = read_pdf(uploaded_resume)
        if pdf_text:
            resume_content = pdf_text
        else:
            st.warning("Could not read the uploaded PDF resume.")
            if default_resume_content:
                st.info("Using the default resume.")
            else:
                st.error("No resume content available.")
                return
    else:
        if default_resume_content:
            st.info("Using the default resume. You can upload your resume in PDF format to use it instead.")
        else:
            st.error("No default resume found. Please upload your resume.")
            return

    # Option to generate email or cover letter
    generation_type = st.radio(
        "Generate:",
        ("Email", "Cover Letter")
    )

    if st.button("Generate"):
        if not name:
            st.error("Please enter your name.")
        elif not email:
            st.error("Please enter your email address.")
        elif not job_description:
            st.error("Please enter the job description.")
        elif not resume_content:
            st.error("No resume content available. Please upload a resume or ensure the default resume PDF exists and is readable.")
        else:
            with st.spinner(f"Generating {generation_type.lower()}..."):
                output_text = generate_application_text(
                    name=name,
                    email=email,
                    job_description=job_description,
                    resume_content=resume_content,
                    generation_type=generation_type,
                    google_api_key=google_api_key,
                )
                st.subheader(f"Generated {generation_type}:")
                st.markdown(output_text)

if __name__ == "__main__":
    lab2()