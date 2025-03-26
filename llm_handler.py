import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse # For streaming type hint
import asyncio
import json
import re # For parsing and highlighting
import logging
# Make sure 'Union' is included in this line:
from typing import Optional, Dict, List, Tuple, Any, AsyncGenerator, Union

# --- Constants ---
GENERATION_MODEL_NAME = 'gemini-1.5-flash' # Model for main content generation
EXTRACTION_MODEL_NAME = 'gemini-1.5-flash' # Model for extraction/validation/parsing/ranking

# Generation Types
TYPE_RESUME = "RESUME"
TYPE_COVER_LETTER = "COVER_LETTER"
TYPE_EMAIL = "EMAIL"

# Email Recipient Types
RECIPIENT_TA_HM = "Talent Acquisition / Hiring Manager"
RECIPIENT_GENERAL = "General Application / Unspecified"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper: Configure Gemini Client ---
gemini_configured = False

async def _configure_gemini(api_key: str):
    """Configures the GenAI client if not already done."""
    global gemini_configured
    if not api_key:
         logging.error("API Key is missing.")
         raise ValueError("Google API Key not provided.")
    if not gemini_configured:
        try:
            genai.configure(api_key=api_key)
            gemini_configured = True
            logging.info("Google GenAI configured.")
        except Exception as e:
            gemini_configured = False # Reset flag on error
            logging.error(f"Failed to configure Google GenAI: {e}", exc_info=True)
            raise ConnectionError(f"Failed to configure Google API: {e}")

# --- Helper: Robust LLM Call (Async) ---
async def _call_llm_async(
    prompt: str,
    api_key: str,
    model_name: str,
    temperature: float = 0.5, # Default temperature
    request_json: bool = False,
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]: # Ensure Union is imported from typing
    """General purpose async LLM caller with error handling."""
    await _configure_gemini(api_key) # Ensure configured

    gen_config = genai.GenerationConfig(temperature=temperature)
    safety_settings=[ # Default safety settings
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    if request_json:
        if hasattr(gen_config, 'response_mime_type'):
            gen_config.response_mime_type = "application/json"
        else:
            logging.warning(f"Model config may not support direct JSON mime type setting. Relying on prompt.")

    try:
        model = genai.GenerativeModel(
            model_name,
            generation_config=gen_config,
            safety_settings=safety_settings
        )
        logging.info(f"Calling model '{model_name}' (stream={stream})...")

        if stream:
            async def stream_generator():
                 response_stream = await model.generate_content_async(prompt, stream=True)
                 async for chunk in response_stream:
                     try:
                         if chunk.text: yield chunk.text
                     except ValueError:
                         logging.warning("Stream chunk blocked by safety filter.")
                         yield "\n--- ERROR: Content blocked by safety filter ---\n"
                         break
                     except Exception as e_chunk:
                         logging.error(f"Error processing stream chunk: {e_chunk}")
                         yield f"\n--- ERROR Processing Stream Chunk: {e_chunk} ---\n"
                         break
            return stream_generator()
        else:
            response = await model.generate_content_async(prompt)
            logging.info(f"Model '{model_name}' finished.")
            try:
                 response_text = response.text.strip()
                 if request_json and response_text.startswith("```json"):
                     response_text = re.sub(r"^```json\s*|\s*```$", "", response_text, flags=re.MULTILINE).strip()
                 return response_text
            except ValueError:
                 logging.error("Response blocked by safety filter.")
                 raise ValueError("Response blocked by safety filter.")
            except Exception as e_text:
                 logging.error(f"Error extracting text from response: {e_text}")
                 raise RuntimeError(f"Failed to extract text from response: {e_text}")

    except Exception as e:
        logging.error(f"LLM call to '{model_name}' failed: {e}", exc_info=True)
        raise RuntimeError(f"LLM API call failed: {e}")


# --- Resume Section Parsing (Async LLM) ---
async def parse_resume_sections_llm(resume_content: str, google_api_key: str) -> Dict[str, Any]:
    """Parses resume text into sections using an LLM call (async). Returns dict with data or error."""
    prompt = f"""
    Analyze the following resume text and parse it into distinct logical sections based on common resume structure (e.g., Summary/Objective, Skills, Experience/Work History, Education, Projects, Certifications).
    Return ONLY a valid JSON object where keys are the standardized section titles (use "Summary", "Skills", "Experience", "Education", "Projects", "Certifications", "Other" if applicable) and values are the corresponding text content extracted from the resume.
    If a section is not found, omit the key. Preserve the original text formatting within each section's value.

    Resume Text:
    ---
    {resume_content}
    ---

    JSON Output:
    """
    try:
        response_text = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=EXTRACTION_MODEL_NAME,
            temperature=0.1,
            request_json=True
        )
        # Robust JSON parsing
        try:
            raw_json = response_text.strip().replace('```json', '').replace('```', '').strip()
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json) # Handle trailing commas
            parsed_json = json.loads(raw_json)
            if not isinstance(parsed_json, dict):
                 raise ValueError("LLM did not return a valid JSON dictionary for resume sections.")
        except json.JSONDecodeError as json_err:
             logging.error(f"Failed to parse resume section JSON: {json_err}. Raw response: {response_text}")
             raise ValueError(f"Invalid JSON response for resume sections: {json_err}")

        logging.info("Resume sections parsed successfully.")
        return {"sections": parsed_json, "error": None}
    except Exception as e:
        logging.error(f"Error parsing resume with LLM: {e}")
        return {"sections": {"full_text": resume_content}, "error": f"Resume Parsing Error: {e}"}

# --- Rank JD Requirements (Async LLM) ---
async def _rank_jd_requirements(requirements: List[str], google_api_key: str) -> Dict[str, Any]:
    """Uses an LLM to rank JD requirements (async). Returns dict with data or error."""
    if not requirements:
         return {"ranked_list": [], "error": None}

    req_string = "\n".join(f"- {req}" for req in requirements)

    prompt = f"""
    Analyze the following job requirements extracted from a job description. Identify the most critical skills and responsibilities.
    Return ONLY a valid JSON object with a single key "ranked_list". The value should be a list of strings, containing the original requirement texts ordered from MOST important to LEAST important based on typical hiring priorities.

    Requirements List:
    ---
    {req_string}
    ---

    JSON Output (key: "ranked_list"):
    """
    try:
        response_text = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=EXTRACTION_MODEL_NAME,
            temperature=0.2,
            request_json=True
        )
        # Robust JSON parsing
        try:
            raw_json = response_text.strip().replace('```json', '').replace('```', '').strip()
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json) # Handle trailing commas
            ranked_data = json.loads(raw_json)
            if not isinstance(ranked_data.get("ranked_list"), list):
                 raise ValueError("LLM did not return a valid JSON list under 'ranked_list'.")
        except json.JSONDecodeError as json_err:
             logging.error(f"Failed to parse JD ranking JSON: {json_err}. Raw response: {response_text}")
             raise ValueError(f"Invalid JSON response for JD ranking: {json_err}")

        logging.info("JD Requirements ranked successfully.")
        return {"ranked_list": ranked_data["ranked_list"], "error": None}
    except Exception as e:
        logging.error(f"Error ranking JD requirements: {e}")
        return {"ranked_list": requirements, "error": f"Ranking Error: {e}"}


# --- Structured Data Extraction (Async LLM - Example) ---
async def _extract_structured_data(text: str, google_api_key: str, doc_type: str) -> Dict[str, Any]:
    """Extracts structured data using LLM (async). Returns dict with data or error."""
    schema = {}
    instructions = ""
    if doc_type.lower() == "job description":
        schema = {
            "job_title": "string (Specific job title)",
            "company_name": "string (Company name)",
            "key_skills_requirements": ["string (List of essential skills, tools, qualifications, or keywords)"],
            "location": "string (Job location, if mentioned)",
            "summary": "string (Brief summary or objective of the role)"
        }
        instructions = "Extract the job title, company name, key skills/requirements (as a list of distinct items), location, and a brief role summary."
    else:
        return {"data": {}, "error": f"Extraction schema not defined for type: {doc_type}"}

    prompt = f"""
    Analyze the following text ({doc_type}) and extract the specified information according to the schema.
    Return ONLY a valid JSON object adhering to the schema. Use null or omit keys if information is not found.
    Instructions: {instructions}
    Schema: {json.dumps(schema)}

    Text:
    ---
    {text}
    ---

    JSON Output:
    """
    try:
        response_text = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=EXTRACTION_MODEL_NAME,
            temperature=0.1,
            request_json=True
        )
        # Robust JSON parsing
        try:
             raw_json = response_text.strip().replace('```json', '').replace('```', '').strip()
             raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json) # Handle trailing commas
             extracted_data = json.loads(raw_json)
             if not isinstance(extracted_data, dict):
                 raise ValueError(f"LLM did not return a valid JSON dictionary for {doc_type}.")
        except json.JSONDecodeError as json_err:
             logging.error(f"Failed to parse structured data JSON ({doc_type}): {json_err}. Raw response: {response_text}")
             raise ValueError(f"Invalid JSON response for {doc_type} extraction: {json_err}")

        logging.info(f"Structured data extracted for {doc_type}.")
        # Ensure list format for requirements if present
        if "key_skills_requirements" in extracted_data and not isinstance(extracted_data["key_skills_requirements"], list):
             logging.warning("Extracted key_skills_requirements was not a list, attempting conversion.")
             # Attempt simple conversion if it's a string, otherwise handle as needed
             if isinstance(extracted_data["key_skills_requirements"], str):
                 # Simple split by comma or newline, might need refinement
                 extracted_data["key_skills_requirements"] = [item.strip() for item in re.split(r'[,\n]', extracted_data["key_skills_requirements"]) if item.strip()]
             else:
                 extracted_data["key_skills_requirements"] = [] # Fallback to empty list

        return {"data": extracted_data, "error": None}
    except Exception as e:
        logging.error(f"Error extracting structured data for {doc_type}: {e}")
        return {"data": {}, "error": f"Extraction Error ({doc_type}): {e}"}


# --- Data Preparation (Async) ---
async def _prepare_common_data(job_description: str, resume_content: str, google_api_key: str) -> Dict[str, Any]:
    """Extracts structured data, parses resume, ranks JD (async). Returns dict with results."""
    results = {"error": None, "jd_data": {}, "resume_sections": {"full_text": resume_content}, "ranked_jd_requirements": []} # Default resume sections
    task_errors = []
    try:
        # Concurrently run JD extraction and resume parsing
        tasks = {
            "jd_extraction": _extract_structured_data(job_description, google_api_key, "job description"),
            "resume_parsing": parse_resume_sections_llm(resume_content, google_api_key),
        }
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        result_map = dict(zip(tasks.keys(), task_results))

        # Process JD Extraction results
        jd_res = result_map.get("jd_extraction")
        if isinstance(jd_res, Exception): task_errors.append(f"JD Extraction failed: {jd_res}")
        elif jd_res.get("error"): task_errors.append(f"JD Extraction error: {jd_res['error']}")
        else: results["jd_data"] = jd_res.get("data", {})

        # Process Resume Parsing results
        resume_res = result_map.get("resume_parsing")
        if isinstance(resume_res, Exception): task_errors.append(f"Resume Parsing failed: {resume_res}")
        elif resume_res.get("error"):
            task_errors.append(f"Resume Parsing error: {resume_res['error']}")
            # Keep the default full_text section already set
        else: results["resume_sections"] = resume_res.get("sections", {"full_text": resume_content})

        # Rank JD requirements if extraction was successful and requirements exist
        extracted_requirements = results.get("jd_data", {}).get("key_skills_requirements", [])
        if extracted_requirements and isinstance(extracted_requirements, list):
            ranking_res = await _rank_jd_requirements(extracted_requirements, google_api_key)
            if isinstance(ranking_res, Exception): task_errors.append(f"JD Ranking failed: {ranking_res}")
            elif ranking_res.get("error"): task_errors.append(f"JD Ranking error: {ranking_res['error']}")
            else: results["ranked_jd_requirements"] = ranking_res.get("ranked_list", [])
        elif results.get("jd_data"): # JD data exists but no requirements found/extracted
             results["ranked_jd_requirements"] = []
             logging.info("No requirements found or extracted from JD to rank.")
        else: # Cannot rank if JD extraction failed
            results["ranked_jd_requirements"] = []
            logging.warning("JD Extraction failed, skipping requirement ranking.")


        if task_errors:
            results["error"] = "; ".join(task_errors)

        results["raw_resume"] = resume_content # Keep raw resume text accessible

    except Exception as e:
        results["error"] = f"Unexpected Error in Data Preparation: {e}"
        logging.error(f"Critical error in _prepare_common_data: {e}", exc_info=True)

    return results


# --- Base Generator with Multi-Turn Refinement (Async Stream) ---
async def generate_application_text_streamed(
    name: str, email: str, job_description: str, resume_content: str,
    generation_type: str, google_api_key: str, tone: str
) -> AsyncGenerator[str, None]:
    """Generates Resume or Cover Letter using multi-turn refinement (async stream)."""
    common_data = {}
    try:
        # --- 1. Prepare Data (Retrieve from state or recalc - for direct call, recalc) ---
        # NOTE: In app.py, prep data is done once. If called standalone, it would run here.
        # For simplicity here, assume prep_data might need to be recalculated if called directly.
        # In the context of app.py, common_data will be pre-populated.
        # If common_data is not passed, calculate it.
        common_data = await _prepare_common_data(job_description, resume_content, google_api_key)
        if common_data.get("error"):
            yield f"\n--- ERROR during data preparation: {common_data['error']} ---\n"
            return

        if not common_data.get("jd_data") or not common_data.get("resume_sections"):
             yield "\n--- ERROR: Failed to get essential JD or Resume data for generation ---\n"
             return

        # --- Format inputs for prompts ---
        resume_sections_str = "\n\n".join(f"**{sec.upper()}**\n{content}" for sec, content in common_data.get("resume_sections", {}).items())
        ranked_req_str = "\n".join(f"- {req}" for req in common_data.get("ranked_jd_requirements", [])) or "N/A"
        jd_title = common_data.get('jd_data', {}).get('job_title', 'N/A')
        jd_company = common_data.get('jd_data', {}).get('company_name', 'N/A')


        # --- 2. Initial Draft Prompt ---
        draft_prompt = f"""
        **Candidate:** {name} ({email})
        **Target Job:** {jd_title} at {jd_company}
        **Ranked Requirements (Most Important First):**\n{ranked_req_str}
        **Parsed Candidate Resume Sections:**\n{resume_sections_str}
        **Desired Tone:** {tone}
        **Task:** Generate the **FIRST DRAFT** of a {generation_type} tailored to the job description, using evidence from the candidate's resume sections. Focus on addressing the ranked requirements clearly.
        **Format:** {'Use standard ATS-friendly professional resume format (Markdown with clear sections like ## Summary, ## Skills, ## Experience, ## Education). Focus on quantifiable achievements.' if generation_type == TYPE_RESUME else 'Use standard professional cover letter format (Intro, Body paragraphs linking experience to key requirements, Conclusion).'}
        {'For COVER LETTER: Aim for approximately 300-450 words.' if generation_type == TYPE_COVER_LETTER else ''}

        **Output ONLY the {generation_type} draft:**
        """

        # --- 3. Generate Initial Draft (Async) ---
        yield f"--- Generating initial {generation_type} draft... ---\n"
        initial_draft = await _call_llm_async(draft_prompt, google_api_key, GENERATION_MODEL_NAME, temperature=0.6)
        if not initial_draft: raise ValueError("Initial draft generation failed or returned empty.")

        # --- 4. Critique Prompt ---
        critique_prompt = f"""
        **Target Job Requirements (Ranked):**\n{ranked_req_str}
        **Initial {generation_type} Draft:**
        ---
        {initial_draft}
        ---

        **Task:** Critique the initial draft based ONLY on these criteria:
        1.  **Requirement Alignment:** How well does it address the MOST IMPORTANT ranked requirements using specific examples/evidence from the (implied) resume? Are keywords used effectively?
        2.  **Clarity & Conciseness:** Is it clear, well-organized, and to the point?
        3.  **Tone Consistency:** Does it maintain the desired '{tone}' tone?
        4.  **ATS Friendliness (Structure/Keywords):** Does it appear well-structured and use relevant terms? {'For resumes, check for standard sections (Summary, Skills, Experience, Education). Avoid tables/columns.' if generation_type == TYPE_RESUME else ''}

        **Output ONLY the critique points (brief bullet points or short sentences):**
        """

        # --- 5. Generate Critique (Async) ---
        yield f"\n--- Generating critique of the draft... ---\n"
        critique = await _call_llm_async(critique_prompt, google_api_key, EXTRACTION_MODEL_NAME, temperature=0.3)
        if not critique: logging.warning("Critique generation returned empty."); critique = "No critique generated."

        # --- 6. Refinement Prompt ---
        refinement_prompt = f"""
        **Candidate:** {name} ({email})
        **Target Job:** {jd_title} at {jd_company}
        **Ranked Requirements:**\n{ranked_req_str}
        **Parsed Candidate Resume Sections:**\n{resume_sections_str}
        **Desired Tone:** {tone}
        **Initial {generation_type} Draft:**
        ---
        {initial_draft}
        ---
        **Critique of Initial Draft:**
        ---
        {critique}
        ---

        **Task:** Generate the **FINAL, REVISED** {generation_type} by carefully addressing the critique points. Improve requirement alignment, clarity, tone consistency, structure, and keyword usage based on the critique. Ensure the final output directly reflects the improvements suggested.
        {'For COVER LETTER: Maintain an approximate total length of 300-450 words.' if generation_type == TYPE_COVER_LETTER else ''}
        {'For RESUME: Ensure final output is valid, ATS-friendly Markdown with standard sections (e.g., ## Summary, ## Skills, ## Experience using bullet points, ## Education). Focus on action verbs and quantifiable results.' if generation_type == TYPE_RESUME else ''}
        **Output ONLY the final {generation_type}. Do NOT include the critique, draft, section headers about the process, or any other commentary.**
        """

        # --- 7. Generate Final Version (Async Stream) ---
        yield f"\n--- Generating final refined {generation_type}... ---\n"
        final_stream = await _call_llm_async(refinement_prompt, google_api_key, GENERATION_MODEL_NAME, temperature=0.6, stream=True)
        async for chunk in final_stream:
             yield chunk

    except Exception as e:
        error_message = f"\n--- Error during {generation_type} Generation: {e} ---"
        logging.error(f"Error in generate_application_text_streamed: {e}", exc_info=True)
        yield error_message
        # Include debug info if available
        if common_data:
             yield f"\nDebug Info (Data Prep Error): {common_data.get('error') or 'OK'}"


# --- Email Generator + Validator (Async) ---
async def generate_email_and_validate(
    name: str, email: str, job_description: str, resume_content: str,
    google_api_key: str, tone: str, email_recipient_type: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Generates email and performs ATS validation (async)."""
    generated_email = None
    validation_results = None
    common_data = {}
    try:
        # --- 1. Prepare Data ---
        # NOTE: Assumes this might be called directly or data is passed via context.
        # Recalculate if needed. In app.py context, data is prepared once.
        common_data = await _prepare_common_data(job_description, resume_content, google_api_key)
        if common_data.get("error"):
            return None, {"error": f"Data Preparation Error: {common_data['error']}"}
        if not common_data.get("jd_data"):
             return None, {"error": "Failed to get JD data during preparation."}

        # --- Format inputs for prompt ---
        resume_sections_str = "\n\n".join(f"**{sec.upper()}**\n{content}" for sec, content in common_data.get("resume_sections", {}).items())
        ranked_req_str = "\n".join(f"- {req}" for req in common_data.get("ranked_jd_requirements", [])) or "N/A"
        jd_title = common_data.get('jd_data', {}).get('job_title', 'N/A')
        jd_company = common_data.get('jd_data', {}).get('company_name', 'N/A')

        # --- 2. Generate Email (Async) ---
        email_prompt = f"""
        **Candidate:** {name} ({email})
        **Target Job:** {jd_title} at {jd_company}
        **Ranked Job Requirements:**\n{ranked_req_str}
        **Parsed Candidate Resume Sections:**\n{resume_sections_str}
        **Desired Tone:** {tone}
        **Email Recipient:** {email_recipient_type}

        **Task:** Generate a concise and professional email (target 150-200 words) for the candidate regarding this job application.
        * Create a clear, specific subject line including the job title.
        * Briefly introduce the candidate and the role they are applying for.
        * Highlight 1-2 key qualifications directly relevant to the MOST IMPORTANT job requirements, referencing their experience from the resume sections provided. Be specific.
        * Express genuine enthusiasm for the role and the company.
        * State that their resume/CV is attached.
        * Keep the tone professional and consistent with '{tone}'.
        * Ensure the language is clear and ATS-friendly (uses relevant keywords naturally).

        **Output ONLY the email content formatted exactly like this:**
        Subject: [Your Subject Line Here]

        [Body of the email starts here]
        """
        logging.info("Generating Email...")
        generated_email = await _call_llm_async(email_prompt, google_api_key, GENERATION_MODEL_NAME, temperature=0.7)
        if not generated_email: raise ValueError("Email generation failed or returned empty.")


        # --- 3. Validate Email (Async) ---
        logging.info("Validating generated Email...")
        validation_results = await _validate_ats_friendliness(
            document_text=generated_email,
            document_type="Email",
            job_description_data=common_data.get("jd_data", {}),
            google_api_key=google_api_key
        )

        return generated_email, validation_results

    except Exception as e:
        logging.error(f"Error in generate_email_and_validate: {e}", exc_info=True)
        error_detail = {"error": f"Email Generation/Validation Error: {e}"}
        if common_data: # Add debug context if prep step ran
            error_detail["debug_prep_error"] = common_data.get("error")
        return None, error_detail


# --- ATS Validator (Async) ---
async def _validate_ats_friendliness(
    document_text: str, document_type: str, job_description_data: Dict, google_api_key: str
) -> Dict[str, Any]:
    """Uses an LLM to evaluate ATS friendliness (async). Returns validation dict."""
    results = {}
    if not document_text:
        return {"error": "No document text provided for validation."}
    try:
        jd_keywords_list = job_description_data.get("key_skills_requirements", [])
        jd_keywords_str = ", ".join(jd_keywords_list) if jd_keywords_list else "N/A"
        jd_summary = job_description_data.get("summary", "N/A")
        jd_title = job_description_data.get("job_title", "N/A")

        prompt = f"""
        **Task:** Evaluate the ATS (Applicant Tracking System) friendliness of the following '{document_type}' intended for the job '{jd_title}'.
        **Job Description Keywords/Requirements:** {jd_keywords_str}
        **Job Description Summary:** {jd_summary}

        **Document Text to Evaluate:**
        ---
        {document_text}
        ---

        **Evaluation Criteria & Output Format:**
        Return ONLY a valid JSON object with the following keys:
        1.  `ats_score` (integer): Score from 1 (Poor) to 5 (Excellent) for overall ATS compatibility, considering keywords, structure, and clarity.
        2.  `keyword_check` (object): Contains:
            * `found_keywords` (list): List of important keywords/skills from the JD found in the text. Limit to top 5-7 relevant ones.
            * `missing_suggestions` (list): List of important keywords/skills from the JD seemingly missing or underrepresented. Limit to top 3-5 actionable suggestions.
            * `density_impression` (string): Qualitative assessment of keyword usage (e.g., "Good density and relevance", "Fair, could add more specifics", "Low density, missing key terms").
        3.  `clarity_structure_check` (string): Brief assessment of clarity and organization for parsing (e.g., "Clear structure, easy to parse", "Generally clear, some long sentences", "Lacks clear sections/paragraphs"). For Resumes, mention if standard sections seem present.
        4.  `formatting_check` (string): Brief assessment of ATS suitability regarding formatting (e.g., "Standard text format, looks clean", "Check for non-standard characters or complex layouts if applicable"). For Resumes, note use of bullet points vs complex tables.
        5.  `overall_feedback` (string): Brief (1-2 sentences), actionable feedback focused on improving ATS compatibility for *this specific document type*.
        """
        response_text = await _call_llm_async(
             prompt,
             google_api_key,
             EXTRACTION_MODEL_NAME,
             temperature=0.2,
             request_json=True
        )

        # Robust JSON parsing
        try:
            raw_json = response_text.strip().replace('```json', '').replace('```', '').strip()
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json) # Handle trailing commas
            results = json.loads(raw_json)
            required_keys = ["ats_score", "keyword_check", "clarity_structure_check", "formatting_check", "overall_feedback"]
            if not all(key in results for key in required_keys):
                 raise ValueError("LLM validation response missing required keys.")
            # Further validate nested structure if needed
            if not isinstance(results.get("keyword_check"), dict) or not all(k in results["keyword_check"] for k in ["found_keywords", "missing_suggestions", "density_impression"]):
                 raise ValueError("LLM validation response keyword_check structure is invalid.")

            logging.info(f"ATS Validation successful for {document_type}.")

        except json.JSONDecodeError as json_err:
            results = {"error": f"Failed to parse LLM validation response as JSON: {json_err}", "raw_response": response_text}
        except ValueError as val_err:
             results = {"error": f"Invalid structure in LLM validation response: {val_err}", "raw_response": response_text}
        except Exception as parse_err:
             results = {"error": f"Error processing LLM validation response: {parse_err}", "raw_response": response_text}

    except Exception as e:
        logging.error(f"Error during ATS validation LLM call for {document_type}: {e}", exc_info=True)
        results = {"error": f"ATS Validation LLM Call Error: {e}"}

    return results