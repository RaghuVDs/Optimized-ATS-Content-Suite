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
    tone: str,
    email_recipient_type: Optional[str] = None
) -> Optional[str]:
    """Generates an ATS-friendly email or cover letter using Google Gemini with a specified tone and recipient type."""
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

    ats_instructions = "Ensure the generated content is optimized for Applicant Tracking Systems (ATS). Use clear and concise language, standard section headings (if applicable), and incorporate relevant keywords from the job description. Avoid complex formatting."

    if generation_type == "Email":
        if email_recipient_type == "Talent Acquisition and Hiring Manager":
            prompt = f"""You are an expert at writing highly effective job application emails tailored for both Talent Acquisition Specialists and Hiring Managers. Your goal is to create a compelling email that highlights the candidate's most relevant qualifications based on their resume and the job description, optimized for Applicant Tracking Systems (ATS).

Candidate's Name: {name}
Candidate's Email: {email}
Candidate's Resume:
{resume_content}
Job Description:
{job_description}
Desired Tone: {tone}

Instructions:

Subject: Highly Qualified [Your Most Relevant Skill/Profession] for [Specific Job Title] - [Your Name]

Dear [Hiring Manager or Talent Acquisition Specialist Name],

I am writing to express my enthusiastic interest in the [Specific Job Title] position at [Company Name], as advertised [mention where you saw the advertisement if applicable]. With my proven experience in [mention 2-3 most relevant skills or areas of expertise directly from your resume that align with the job description], I am confident I possess the qualifications to significantly contribute to your team.

In my previous role as a [Your Previous Role] at [Your Previous Company], I [describe a specific, quantifiable achievement or responsibility from your resume that directly addresses a key requirement in the job description. Use action verbs and numbers whenever possible. For example, "Led a project that resulted in a X% increase in Y" or "Successfully implemented a solution that reduced Z cost by W%"]. This experience, coupled with my skills in [mention 2-3 other relevant skills from your resume, including technical skills like 'Azure Data Factory', 'PySpark', 'Tableau', etc.], makes me a strong fit for this opportunity.

I am particularly drawn to [Company Name]'s work in [mention something specific about the company or the role that excites you, demonstrating you've done your research]. My experience in [mention another relevant skill or project from your resume, such as 'Data Pipeline and Analysis with Azure Data Factory' or 'Airfare Prediction and Optimization with PySpark'] aligns well with the requirements outlined in the job description, and I am eager to bring my expertise to your organization.

Thank you for considering my application. My resume, attached for your convenience, provides further details on my qualifications. I am available for an interview at your earliest convenience.

Sincerely,
{name}
[Your Phone Number (if available in resume)]
[Your LinkedIn Profile URL (if available in resume)]
{email}

Key Considerations for ATS Optimization:

- Use clear and concise language.
- Incorporate relevant keywords from the job description naturally throughout the email.
- Avoid complex formatting or graphics.
- Use standard professional fonts (though this is less of a concern for email).

Please ensure the email is professional, enthusiastic, and directly addresses the key requirements of the job description, highlighting the most relevant skills and achievements from the candidate's resume.
"""
        elif email_recipient_type == "General Employer":
            prompt = f"""You are an expert at writing compelling job application emails suitable for a broad range of employers. Your goal is to create a professional and engaging email based on the candidate's resume and the job description, optimized for Applicant Tracking Systems (ATS).

Candidate's Name: {name}
Candidate's Email: {email}
Candidate's Resume:
{resume_content}
Job Description:
{job_description}
Desired Tone: {tone}

Instructions:

Subject: Application for [Specific Job Title] - [Your Name]

Dear [Hiring Manager],

I am writing to express my interest in the [Specific Job Title] position at [Company Name]. With a background in [mention your primary field or expertise from your resume] and a strong foundation in [mention 2-3 key skills relevant to the job description], I am confident in my ability to contribute to your team's success.

During my time at [Mention a relevant company from your resume], I gained valuable experience in [mention 2-3 key responsibilities or skills developed in that role that align with the job description]. I am particularly skilled in [mention a specific achievement or area of strength from your resume, such as 'designed and implemented data solutions', 'developed and optimized ETL pipelines', or 'applied ML techniques'].

I am eager to learn more about the opportunities at [Company Name] and how my skills in [mention another relevant skill or technology from your resume, like 'Azure', 'Databricks', 'Tableau', 'PyTorch', etc.] can support your objectives. My resume, which I have attached, provides further detail on my qualifications and experience.

Thank you for your time and consideration. I look forward to the possibility of discussing this opportunity further.

Sincerely,
{name}
[Your Phone Number (if available in resume)]
[Your LinkedIn Profile URL (if available in resume)]
{email}

Key Considerations for ATS Optimization:

- Use clear and straightforward language.
- Incorporate keywords from the job description where appropriate.
- Maintain a professional and standard email format.

Please ensure the email is professional and highlights the candidate's relevant skills and experience for a general employer.
"""
        else:
            prompt = f"""You are an expert at writing job application emails. Based on the information provided in the user's resume and the job description, write an email in the following format:

Subject: Experienced [Your Profession] for [Position] - [Your Name]

Dear [Recruiter's Name],

I hope you're well. My name is {name}, and I am an experienced [Your Profession/Current Role] with expertise in [Key Skills/Industries]. I am reaching out to express my interest in [Position or type of role] at [Company Name] because I believe my background in [specific skills/experiences] aligns well with your team's needs.

In my previous role at [Current or Previous Company], I [briefly mention a key achievement or responsibility]. I am confident that my experience in [relevant skills] would make a significant contribution to [Company Name].

I would welcome the opportunity to discuss how my skills and experiences can benefit your organization. Are you available for a brief call next week? Please let me know a time that works for you, or feel free to suggest an alternative.

Thank you for your time and consideration.

Best regards,
{name}
[Your Phone Number]
[LinkedIn Profile]
{email}

Use the following information to fill in the bracketed placeholders:

Your Name: {name}
Your Email: {email}
Your Resume: {resume_content}
Job Description: {job_description}

Instructions:

- **[Your Profession]**: Identify the user's primary profession or most relevant role from their resume.
- **[Position]**: Extract the specific job title from the Job Description.
- **[Recruiter's Name]**: If the recruiter's name is mentioned in the Job Description, use it. Otherwise, use a generic salutation like "Hiring Manager,".
- **[Your Profession/Current Role]**: Similar to "[Your Profession]", use the user's current or most relevant role from the resume.
- **[Key Skills/Industries]**: Identify 2-3 key skills or industries from the user's resume that are highly relevant to the requirements mentioned in the Job Description.
- **[Position or type of role]**: Use the specific job title from the Job Description or a slightly broader term if appropriate.
- **[Company Name]**: Extract the name of the company from the Job Description.
- **[specific skills/experiences]**: Mention 1-2 specific skills or experiences from the user's resume that directly address requirements or preferences stated in the Job Description.
- **[Current or Previous Company]**: Identify the user's most recent or a highly relevant previous company from their resume.
- **[briefly mention a key achievement or responsibility]**: Summarize a significant achievement or key responsibility from the user's resume at the mentioned company that demonstrates their ability to perform well in the target role (based on the Job Description). Keep it concise.
- **[relevant skills]**: Mention 1-2 more relevant skills from the user's resume that align with the Job Description.
- **[Your Phone Number]**: If available in the resume, use it. Otherwise, you can use a placeholder like "[Your Phone Number]".
- **[LinkedIn Profile]**: If available in the resume, use it. Otherwise, you can use a placeholder like "[LinkedIn Profile]".

Ensure the generated email is professional, enthusiastic, and tailored to the specific job.
"""
    elif generation_type == "Cover Letter":
        prompt = f"""You are an expert at writing highly effective cover letters for data professionals. Your goal is to create a compelling and tailored cover letter that will impress hiring managers. Use the information provided in the user's resume and the job description to craft the best possible cover letter.

        User's Name: {name}
        User's Email: {email}
        User's Resume: {resume_content}
        Job Description: {job_description}
        Desired Tone: {tone}

        Instructions for writing the cover letter:

        1.  **Start with a Strong Opening:** Clearly state the specific position you are applying for at the beginning of the letter. Express your strong interest in the role and the company, mentioning where you saw the job posting if applicable.

        2.  **Highlight Key Skills and Expertise:** Based on the Job Description, identify the most critical skills and qualifications being sought. Then, draw specific examples and evidence from the user's resume that demonstrate your proficiency in these areas. Focus on data-related skills such as:
            * Data Analysis
            * Statistical Modeling
            * Machine Learning (if applicable)
            * Data Visualization
            * Database Management (SQL, NoSQL, etc.)
            * Programming Languages (Python, R, etc.)
            * Big Data technologies (Spark, Hadoop, etc., if relevant)
            * Cloud platforms (AWS, Azure, GCP, if relevant)

        3.  **Showcase Quantifiable Achievements:** Instead of just listing skills, describe specific projects or accomplishments from the user's resume where they utilized these skills to achieve measurable results. Quantify your impact whenever possible (e.g., "Improved model accuracy by X%", "Reduced data processing time by Y", "Generated insights that led to Z increase in...").

        4.  **Directly Address Job Requirements:** Go through the Job Description point by point and explicitly address how your skills and experience align with each requirement and responsibility mentioned. Use keywords and phrases from the job description in your cover letter.

        5.  **Demonstrate Understanding of the Company:** Briefly show that you have researched the company and understand its mission, values, or recent work, especially in the data science or analytics domain. Explain why you are particularly interested in this company.

        6.  **Maintain a Professional and Enthusiastic Tone:** Write in a formal yet enthusiastic and confident tone. Proofread carefully for any grammatical errors or typos.

        7.  **Structure the Cover Letter Logically:** Follow a standard cover letter format:
            * Your Contact Information (Name, Email - Phone and LinkedIn can be included if present in the resume)
            * Date
            * Hiring Manager's Name (if known, otherwise use "Hiring Manager")
            * Hiring Manager's Title
            * Company Name
            * Company Address (optional)
            * Salutation (e.g., "Dear [Hiring Manager Name],")
            * Opening Paragraph (Position and interest)
            * Body Paragraphs (Skills, Experience, Achievements, Alignment with Job)
            * Company Interest Paragraph
            * Closing Paragraph (Express interest in an interview and thank them for their time)
            * Professional Closing (e.g., "Sincerely,")
            * Your Name

        8.  **Call to Action:** In your closing paragraph, express your eagerness to learn more about the opportunity and discuss how your skills and experience can contribute to the company's success.

        Write a cover letter that is approximately 1 page in length and is highly persuasive.
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

    prompt = f"""Craft a highly tailored and impactful resume that positions you as the ideal candidate for the role, using the default resume and job description provided. Ensure the resume is rigorously optimized for ATS by integrating targeted keywords, clear formatting, and emphasis on the most relevant skills and achievements.

        Default Resume:
        {default_resume}

        Job Description:
        {job_description}

        Key Focus Areas:
        - Revise the summary to directly reflect and align with the job description.
        - Emphasize pertinent skills, achievements, and experiences that demonstrate your fit for the role.
        - Strategically incorporate keywords from the job description to maximize ATS compatibility.
        - Maintain a professional, industry-standard resume format.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating tailored resume: {e}"