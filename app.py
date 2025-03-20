# app.py
import streamlit as st
from typing import Optional
from pdf_handler import read_pdf
from llm_handler import generate_application_text, generate_tailored_resume
import os

def job():
    st.markdown(
        "<h1 style='text-align: center;'>📧📄 Automated Job Application Text Generation</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Enter your details and the job description to generate an email, cover letter, or a tailored resume.</p>",
        unsafe_allow_html=True,
    )

    # API key handling for Google Gemini
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API key not found in secrets.")
        st.stop()

    # Sidebar for options
    with st.sidebar:
        st.subheader("Generation Options")
        tone = st.selectbox(
            "Choose the Tone:",
            ("Formal", "Friendly", "Enthusiastic", "Creative", "Concise")
        )

    # Input fields
    name = st.text_input("Your Name:")
    email = st.text_input("Your Email Address:")
    job_description = st.text_area("Job Description:")

    # Option to upload resume
    uploaded_resume = st.file_uploader("Upload your Resume (PDF only):", type=["pdf"])

    # Load default resume from PDF
    default_resume_content: Optional[str] = None
    try:
        with open("Raghuveera_N_resume.pdf", "rb") as default_resume_file:
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

    # Option to generate tailored resume
    generate_resume = st.checkbox("Generate a tailored resume based on the job description?")
    tailored_resume_output: Optional[str] = None
    if generate_resume and default_resume_content and job_description:
        with st.spinner("Generating tailored resume..."):
            tailored_resume_output = generate_tailored_resume(default_resume_content, job_description, google_api_key)
        if tailored_resume_output:
            st.subheader("Generated Tailored Resume:")
            st.markdown(tailored_resume_output)
            resume_content = tailored_resume_output # Use the generated resume for further steps
        else:
            st.error("Failed to generate tailored resume.")

    # Option to generate email or cover letter
    st.subheader("Generate Application Text")
    generation_type = st.radio(
        "Choose to generate:",
        ("Email", "Cover Letter")
    )

    email_recipient_type: Optional[str] = None
    if generation_type == "Email":
        email_recipient_type = st.radio(
            "Send email to:",
            ("Talent Acquisition and Hiring Manager", "General Employer")
        )

    if st.button("Generate Application Text"):
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
                    tone=tone,
                    email_recipient_type=email_recipient_type # Pass the new parameter
                )
                st.subheader(f"Generated {generation_type} ({tone} Tone):")
                st.markdown(output_text)

                # Feedback section
                st.subheader("Feedback")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Good"):
                        print("Feedback: Positive")  # In a real app, store this feedback
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 Bad"):
                        print("Feedback: Negative")  # In a real app, store this feedback
                        st.error("Thank you for your feedback. We'll work on improving.")

if __name__ == "__main__":
    job()