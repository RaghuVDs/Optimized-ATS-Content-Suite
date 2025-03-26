# llm_handler.py

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [LLM_Handler] %(message)s')

# --- Helper: Configure Gemini Client ---
gemini_configured = False
gemini_config_lock = asyncio.Lock() # Lock to prevent race conditions during config

async def _configure_gemini(api_key: str):
    """Configures the GenAI client if not already done, thread-safe."""
    global gemini_configured
    if not gemini_configured:
        async with gemini_config_lock:
            # Double check after acquiring lock
            if not gemini_configured:
                if not api_key:
                    logging.error("API Key is missing.")
                    raise ValueError("Google API Key not provided.")
                try:
                    genai.configure(api_key=api_key)
                    gemini_configured = True
                    logging.info("Google GenAI configured.")
                except Exception as e:
                    gemini_configured = False # Ensure flag is reset on error
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
) -> Union[str, AsyncGenerator[str, None]]:
    """General purpose async LLM caller with error handling and safety defaults."""
    await _configure_gemini(api_key) # Ensure configured

    gen_config = genai.GenerationConfig(temperature=temperature)
    # Default safety settings - blocking potentially harmful content
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    if request_json:
        # Attempt to set response MIME type for models supporting it
        if hasattr(gen_config, 'response_mime_type'):
            gen_config.response_mime_type = "application/json"
            logging.info(f"Requesting JSON output via mime type for model '{model_name}'.")
        else:
            logging.warning(f"Model config for '{model_name}' may not support direct JSON mime type setting. Relying on prompt instructions for JSON format.")

    try:
        model = genai.GenerativeModel(
            model_name,
            generation_config=gen_config,
            safety_settings=safety_settings
        )
        logging.info(f"Calling model '{model_name}' (stream={stream})...")

        if stream:
            # Define and return the async generator for streaming
            async def stream_generator():
                 try:
                     response_stream = await model.generate_content_async(prompt, stream=True)
                     async for chunk in response_stream:
                         try:
                             # Check for text and potential blocking in the chunk
                             if hasattr(chunk, 'text') and chunk.text:
                                 yield chunk.text
                             elif chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                                 logging.warning(f"Stream chunk blocked by safety filter: {chunk.prompt_feedback.block_reason}")
                                 yield f"\n--- ERROR: Content blocked by safety filter ({chunk.prompt_feedback.block_reason}) ---\n"
                                 break # Stop streaming if blocked
                         except ValueError as val_err: # Often indicates blocked content access
                             logging.warning(f"Stream chunk ValueError (likely safety block): {val_err}")
                             yield f"\n--- ERROR: Content blocked by safety filter ---\n"
                             break
                         except Exception as e_chunk:
                             logging.error(f"Error processing stream chunk: {e_chunk}")
                             yield f"\n--- ERROR Processing Stream Chunk: {e_chunk} ---\n"
                             break
                 except Exception as stream_err:
                      logging.error(f"Error initiating or processing stream for model '{model_name}': {stream_err}", exc_info=True)
                      yield f"\n--- ERROR DURING STREAMING: {stream_err} ---\n"

            return stream_generator()
        else:
            # Non-streaming call
            response = await model.generate_content_async(prompt)
            logging.info(f"Model '{model_name}' finished.")

            # Check for blocked response before accessing text
            if not response.candidates:
                 block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                 logging.error(f"Response blocked by safety filter (reason: {block_reason}). Prompt Feedback: {response.prompt_feedback}")
                 raise ValueError(f"Response blocked by safety filter (reason: {block_reason})")

            # Access text safely
            try:
                 response_text = response.text.strip()
                 # Clean JSON if requested
                 if request_json and response_text.startswith("```json"):
                     response_text = re.sub(r"^```json\s*|\s*```$", "", response_text, flags=re.MULTILINE).strip()
                 return response_text
            except ValueError as val_err: # Sometimes text access raises ValueError if blocked
                 logging.error(f"Response text access error (likely safety block): {val_err}")
                 raise ValueError(f"Response blocked by safety filter (access error): {val_err}")
            except Exception as e_text:
                 logging.error(f"Error extracting text from response: {e_text}")
                 raise RuntimeError(f"Failed to extract text from response: {e_text}")

    except Exception as e:
        logging.error(f"LLM call to '{model_name}' failed: {e}", exc_info=True)
        # Re-raise a more informative error
        raise RuntimeError(f"LLM API call failed for model '{model_name}': {e}")


# --- Resume Section & Keyword Parsing (Async LLM) ---
async def parse_resume_sections_and_keywords_llm(resume_content: str, google_api_key: str) -> Dict[str, Any]:
    """
    Parses resume text into sections AND extracts keywords/skills using LLM (async).
    Returns dict containing 'sections', 'extracted_keywords', and 'error' (None on success).
    """
    prompt = f"""
    Analyze the following resume text. Perform two tasks:
    1. Parse it into distinct logical sections (e.g., Summary/Objective, Skills, Experience/Work History, Education, Projects). Use standardized keys ("Summary", "Skills", "Experience", "Education", "Projects", "Certifications", "Other"). Preserve original text within sections.
    2. Extract a comprehensive list of specific skills, technologies, tools, and methodologies mentioned throughout the resume.

    Return ONLY a valid JSON object with two top-level keys:
    - "sections": An object containing the parsed section texts (e.g., {{"Summary": "...", "Skills": "..."}}). Omit keys for missing sections.
    - "extracted_keywords": A list of unique strings representing the extracted skills/keywords (e.g., ["Python", "Project Management", "AWS", "Agile", "SQL"]). Normalize keywords to lowercase.

    Resume Text:
    ---
    {resume_content}
    ---

    JSON Output:
    """
    # Default structure in case of errors
    error_result = {"sections": {"full_text": resume_content}, "extracted_keywords": [], "error": "Initialization error"}
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

            # Validate structure and sanitize keywords
            if not isinstance(parsed_json, dict):
                raise ValueError("LLM did not return a dictionary.")

            sections = parsed_json.get("sections")
            if not isinstance(sections, dict):
                 logging.warning("LLM did not return valid 'sections' dictionary. Using full text.")
                 sections = {"full_text": resume_content} # Fallback

            keywords_raw = parsed_json.get("extracted_keywords")
            if not isinstance(keywords_raw, list):
                 logging.warning("LLM did not return valid 'extracted_keywords' list. Using empty list.")
                 keywords = []
            else:
                # Ensure strings, lowercase, unique
                keywords = sorted(list(set([str(kw).lower().strip() for kw in keywords_raw if str(kw).strip()])))

            logging.info("Resume sections and keywords parsed successfully.")
            return {"sections": sections, "extracted_keywords": keywords, "error": None}

        except json.JSONDecodeError as json_err:
             logging.error(f"Failed to parse resume section/keyword JSON: {json_err}. Raw response: {response_text[:500]}...")
             error_result["error"] = f"Invalid JSON response for resume parsing: {json_err}"
             return error_result
        except ValueError as val_err:
             logging.error(f"Validation error for resume parsing JSON: {val_err}. Raw response: {response_text[:500]}...")
             error_result["error"] = f"LLM response structure error: {val_err}"
             # Keep existing fallback structure in error_result
             return error_result

    except Exception as e:
        logging.error(f"Error parsing resume sections/keywords with LLM: {e}", exc_info=True)
        error_result["error"] = f"Resume Parsing/Keyword Extraction Error: {e}"
        return error_result


# --- Rank JD Requirements (Async LLM) ---
async def _rank_jd_requirements(requirements: List[str], google_api_key: str) -> Dict[str, Any]:
    """Uses an LLM to rank JD requirements (async). Returns dict with data or error."""
    if not requirements:
         return {"ranked_list": [], "error": None}

    req_string = "\n".join(f"- {req.strip()}" for req in requirements if req.strip())
    if not req_string: # Handle case where input list had only empty strings
         return {"ranked_list": [], "error": None}

    prompt = f"""
    Analyze the following job requirements extracted from a job description. Identify the most critical skills, experiences, and qualifications likely essential for the role.
    Return ONLY a valid JSON object with a single key "ranked_list". The value should be a list of strings, containing the original requirement texts ordered from MOST important to LEAST important based on typical hiring priorities (e.g., mandatory skills > preferred skills > general responsibilities).

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
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)
            ranked_data = json.loads(raw_json)
            if not isinstance(ranked_data.get("ranked_list"), list):
                 raise ValueError("LLM did not return a valid JSON list under 'ranked_list'.")
        except json.JSONDecodeError as json_err:
             logging.error(f"Failed to parse JD ranking JSON: {json_err}. Raw response: {response_text[:500]}...")
             raise ValueError(f"Invalid JSON response for JD ranking: {json_err}")

        logging.info("JD Requirements ranked successfully.")
        return {"ranked_list": ranked_data["ranked_list"], "error": None}
    except Exception as e:
        logging.error(f"Error ranking JD requirements: {e}", exc_info=True)
        # Return original list on error, but flag it
        return {"ranked_list": requirements, "error": f"Ranking Error: {e}"}


# --- Structured Data Extraction (Async LLM - Example for JD) ---
async def _extract_structured_data(text: str, google_api_key: str, doc_type: str) -> Dict[str, Any]:
    """Extracts structured data using LLM (async). Returns dict with 'data' or 'error'."""
    schema = {}
    instructions = ""
    if doc_type.lower() == "job description":
        schema = {
            "job_title": "string (Specific job title, e.g., 'Senior Software Engineer')",
            "company_name": "string (Company name)",
            "key_skills_requirements": ["string (List of distinct, essential technical skills, tools, qualifications, e.g., 'Python', 'AWS', 'Machine Learning', '5+ years experience')"],
            "location": "string (Job location, e.g., 'Remote', 'New York, NY')",
            "summary": "string (Brief 1-2 sentence summary or objective of the role)"
        }
        instructions = "Extract the specific job title, company name, a list of key skills/technologies/experience requirements, location, and a brief role summary."
    else:
        return {"data": {}, "error": f"Extraction schema not defined for type: {doc_type}"}

    prompt = f"""
    Analyze the following text ({doc_type}) and extract the specified information according to the schema.
    Return ONLY a valid JSON object adhering to the schema. Use null or omit keys if information is not found. Be precise.
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
             raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)
             extracted_data = json.loads(raw_json)
             if not isinstance(extracted_data, dict):
                 raise ValueError(f"LLM did not return a valid JSON dictionary for {doc_type}.")
        except json.JSONDecodeError as json_err:
             logging.error(f"Failed to parse structured data JSON ({doc_type}): {json_err}. Raw response: {response_text[:500]}...")
             raise ValueError(f"Invalid JSON response for {doc_type} extraction: {json_err}")

        logging.info(f"Structured data extracted for {doc_type}.")
        # Ensure key_skills_requirements is a list of strings
        reqs = extracted_data.get("key_skills_requirements")
        if reqs is not None and not isinstance(reqs, list):
             logging.warning("Extracted key_skills_requirements was not a list, attempting conversion.")
             if isinstance(reqs, str):
                 extracted_data["key_skills_requirements"] = [item.strip() for item in re.split(r'[,\n;•]', reqs) if item.strip()]
             else:
                 extracted_data["key_skills_requirements"] = []
        elif reqs is None:
             extracted_data["key_skills_requirements"] = [] # Ensure key exists

        return {"data": extracted_data, "error": None}
    except Exception as e:
        logging.error(f"Error extracting structured data for {doc_type}: {e}", exc_info=True)
        return {"data": {}, "error": f"Extraction Error ({doc_type}): {e}"}


# --- Data Preparation (Async) ---
async def _prepare_common_data(job_description: str, resume_content: str, google_api_key: str) -> Dict[str, Any]:
    """Extracts data, parses resume+keywords, ranks JD, finds missing keywords (async)."""
    results = {
        "error": None,
        "jd_data": {},
        "resume_sections": {"full_text": resume_content}, # Default
        "resume_keywords": [],
        "ranked_jd_requirements": [],
        "missing_keywords_from_resume": [] # New field
    }
    task_errors = []
    try:
        # Concurrently run JD extraction and Resume parsing/keyword extraction
        tasks = {
            "jd_extraction": _extract_structured_data(job_description, google_api_key, "job description"),
            "resume_parsing_keywords": parse_resume_sections_and_keywords_llm(resume_content, google_api_key),
        }
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        result_map = dict(zip(tasks.keys(), task_results))

        # Process JD Extraction results
        jd_res = result_map.get("jd_extraction")
        if isinstance(jd_res, Exception): task_errors.append(f"JD Extraction failed: {jd_res}")
        elif jd_res.get("error"): task_errors.append(f"JD Extraction error: {jd_res['error']}")
        else: results["jd_data"] = jd_res.get("data", {})

        # Process Resume Parsing & Keyword results
        resume_res = result_map.get("resume_parsing_keywords")
        if isinstance(resume_res, Exception): task_errors.append(f"Resume Parsing failed: {resume_res}")
        elif resume_res.get("error"):
            task_errors.append(f"Resume Parsing/Keyword error: {resume_res['error']}")
            # Keep the default full_text section
        else:
            results["resume_sections"] = resume_res.get("sections", {"full_text": resume_content})
            results["resume_keywords"] = resume_res.get("extracted_keywords", [])

        # Rank JD requirements if successfully extracted
        extracted_requirements = results.get("jd_data", {}).get("key_skills_requirements", [])
        if extracted_requirements and isinstance(extracted_requirements, list):
            ranking_res = await _rank_jd_requirements(extracted_requirements, google_api_key)
            if isinstance(ranking_res, Exception): task_errors.append(f"JD Ranking failed: {ranking_res}")
            elif ranking_res.get("error"): task_errors.append(f"JD Ranking error: {ranking_res['error']}")
            else: results["ranked_jd_requirements"] = ranking_res.get("ranked_list", [])
        elif results.get("jd_data"):
             results["ranked_jd_requirements"] = []
             logging.info("No requirements found or extracted from JD to rank.")
        else:
            results["ranked_jd_requirements"] = []
            logging.warning("JD Extraction failed, skipping requirement ranking.")

        # --- Identify Missing Keywords ---
        if results.get("jd_data") and "resume_keywords" in results: # Check if resume_keywords extraction was attempted
             try:
                # Normalize JD requirements for comparison
                jd_req_list = results.get("jd_data", {}).get("key_skills_requirements", [])
                jd_keywords_lower = set(str(kw).lower().strip() for kw in jd_req_list if str(kw).strip())
                # Resume keywords are already lowercased during extraction
                resume_keywords_lower = set(results.get("resume_keywords", []))

                missing_keywords = sorted(list(jd_keywords_lower - resume_keywords_lower))
                results["missing_keywords_from_resume"] = missing_keywords
                logging.info(f"Identified potentially missing keywords: {missing_keywords}")
             except Exception as kw_err:
                  task_errors.append(f"Keyword comparison error: {kw_err}")
                  results["missing_keywords_from_resume"] = []
        else:
             results["missing_keywords_from_resume"] = []
             logging.info("Skipping missing keyword identification (JD or resume keywords missing/failed).")
        # -----------------------------

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
    """Generates Resume or Cover Letter using multi-turn refinement, incorporating missing keywords (async stream)."""
    common_data = {}
    try:
        # --- 1. Prepare Data ---
        common_data = await _prepare_common_data(job_description, resume_content, google_api_key)
        if common_data.get("error"):
            yield f"\n--- ERROR during data preparation: {common_data['error']} ---\n"; return
        if not common_data.get("jd_data") or not common_data.get("resume_sections"):
             yield "\n--- ERROR: Failed to get essential JD or Resume data ---\n"; return

        # --- Format inputs ---
        resume_sections_str = "\n\n".join(f"**{sec.upper()}**\n{content}" for sec, content in common_data.get("resume_sections", {}).items() if content) # Only include sections with content
        ranked_req_str = "\n".join(f"- {req}" for req in common_data.get("ranked_jd_requirements", [])) or "N/A"
        jd_title = common_data.get('jd_data', {}).get('job_title', 'N/A')
        jd_company = common_data.get('jd_data', {}).get('company_name', 'N/A')
        missing_keywords = common_data.get("missing_keywords_from_resume", [])
        missing_keywords_str = ", ".join(f"`{kw}`" for kw in missing_keywords) if missing_keywords else "None identified"

        # --- 2. Initial Draft Prompt ---
        draft_prompt = f"""
        **Candidate:** {name} ({email})
        **Target Job:** {jd_title} at {jd_company}
        **Ranked Requirements (Most Important First):**\n{ranked_req_str}
        **Parsed Candidate Resume Sections:**\n{resume_sections_str}
        **Desired Tone:** {tone}
        **Potentially Missing Keywords (from JD):** {missing_keywords_str}

        **Task:** Generate the **FIRST DRAFT** of a {generation_type} tailored to the job description.
        1. Use evidence from the candidate's resume sections to clearly address the ranked requirements.
        2. **Incorporate Missing Keywords:** Where appropriate and supported by the candidate's actual experience described in the resume sections, naturally weave in keywords from the 'Potentially Missing Keywords' list. Rephrase existing achievements or responsibilities if possible. **DO NOT add skills or experiences the candidate doesn't possess.**
        3. Adhere to the desired '{tone}'.
        **Format:** {'Use standard ATS-friendly professional resume format (Markdown with ## Summary, ## Skills, ## Experience, ## Education). Focus on quantifiable results.' if generation_type == TYPE_RESUME else 'Use standard professional cover letter format (Intro, Body linking experience to requirements, Conclusion).'}
        {'For COVER LETTER: Aim for approx 300-450 words.' if generation_type == TYPE_COVER_LETTER else ''}

        **Output ONLY the {generation_type} draft:**
        """

        # --- 3. Generate Initial Draft ---
        yield f"--- Generating initial {generation_type} draft (incorporating keywords)... ---\n"
        initial_draft = await _call_llm_async(draft_prompt, google_api_key, GENERATION_MODEL_NAME, temperature=0.6)
        if not initial_draft: raise ValueError("Initial draft generation failed or returned empty.")

        # --- 4. Critique Prompt ---
        critique_prompt = f"""
        **Target Job Requirements (Ranked):**\n{ranked_req_str}
        **Potentially Missing Keywords Attempted:** {missing_keywords_str}
        **Initial {generation_type} Draft:**
        ---
        {initial_draft}
        ---

        **Task:** Critique the initial draft based ONLY on these criteria:
        1.  **Requirement Alignment & Evidence:** Does it address the MOST IMPORTANT ranked requirements using specific examples?
        2.  **Missing Keyword Integration:** Were any 'Potentially Missing Keywords' incorporated *naturally* and appropriately (without seeming forced/invented)?
        3.  **Clarity & Conciseness:** Is it clear and readable?
        4.  **Tone Consistency:** Does it match the desired '{tone}' tone?
        5.  **ATS Friendliness (Structure/Keywords):** General impression. {'For resumes, check for standard sections.' if generation_type == TYPE_RESUME else ''}

        **Output ONLY the critique points (brief bullet points):**
        """

        # --- 5. Generate Critique ---
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
        **Potentially Missing Keywords List:** {missing_keywords_str}
        **Initial {generation_type} Draft:**
        ---
        {initial_draft}
        ---
        **Critique of Initial Draft:**
        ---
        {critique}
        ---

        **Task:** Generate the **FINAL, REVISED** {generation_type} by meticulously addressing the critique points.
        * Improve requirement alignment, clarity, tone consistency, and structure.
        * **Refine Keyword Integration:** Ensure keywords (especially any 'Potentially Missing Keywords' mentioned in the critique or list) are woven in naturally, accurately reflecting the candidate's likely experience described in the resume sections. **Reiterate: Do NOT invent skills.**
        {'For COVER LETTER: Maintain approx 300-450 words.' if generation_type == TYPE_COVER_LETTER else ''}
        {'For RESUME: Ensure valid, ATS-friendly Markdown with standard sections (e.g., ## Summary, ## Skills, ## Experience using bullet points, ## Education). Use action verbs and quantifiable results.' if generation_type == TYPE_RESUME else ''}
        **Output ONLY the final {generation_type}. No extra commentary.**
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
        if common_data: yield f"\nDebug Info (Data Prep Error): {common_data.get('error') or 'OK'}"


# --- Email Generator + Validator (Async) ---
async def generate_email_and_validate(
    name: str, email: str, job_description: str, resume_content: str,
    google_api_key: str, tone: str, email_recipient_type: str
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Generates email, tries incorporating missing keywords, and validates (async)."""
    generated_email = None
    validation_results = None
    common_data = {}
    try:
        # --- 1. Prepare Data ---
        common_data = await _prepare_common_data(job_description, resume_content, google_api_key)
        if common_data.get("error"): return None, {"error": f"Data Preparation Error: {common_data['error']}"}
        if not common_data.get("jd_data"): return None, {"error": "Failed to get JD data."}

        # --- Format inputs ---
        resume_sections_str = "\n\n".join(f"**{sec.upper()}**\n{content}" for sec, content in common_data.get("resume_sections", {}).items() if content)
        ranked_req_str = "\n".join(f"- {req}" for req in common_data.get("ranked_jd_requirements", [])) or "N/A"
        jd_title = common_data.get('jd_data', {}).get('job_title', 'N/A')
        jd_company = common_data.get('jd_data', {}).get('company_name', 'N/A')
        missing_keywords = common_data.get("missing_keywords_from_resume", [])
        missing_keywords_str = ", ".join(f"`{kw}`" for kw in missing_keywords) if missing_keywords else "None identified"

        # --- 2. Generate Email (Async) ---
        email_prompt = f"""
        **Candidate:** {name} ({email})
        **Target Job:** {jd_title} at {jd_company}
        **Ranked Job Requirements:**\n{ranked_req_str}
        **Parsed Candidate Resume Sections:**\n{resume_sections_str}
        **Desired Tone:** {tone}
        **Email Recipient:** {email_recipient_type}
        **Potentially Missing Keywords (from JD):** {missing_keywords_str}

        **Task:** Generate a concise and professional email (target 150-200 words) for the candidate.
        * Create a clear, specific subject line including the job title.
        * Briefly introduce the candidate and highlight 1-2 key qualifications directly relevant to the MOST IMPORTANT job requirements, using evidence from the resume sections.
        * **Incorporate Keywords:** Where it fits naturally with the candidate's described experience, try to use terms from the 'Potentially Missing Keywords' list. **Do NOT invent skills.**
        * Express enthusiasm. Mention resume attachment. Keep tone professional ('{tone}').
        * Ensure clear language for ATS.

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
        if common_data: error_detail["debug_prep_error"] = common_data.get("error")
        return None, error_detail


# --- ATS Validator (Async) ---
async def _validate_ats_friendliness(
    document_text: str, document_type: str, job_description_data: Dict, google_api_key: str
) -> Dict[str, Any]:
    """Uses an LLM to evaluate ATS friendliness (async). Returns validation dict."""
    results = {"error": None} # Default error state
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
            * `found_keywords` (list): List of important keywords/skills from the JD found in the text. List up to 10 distinct relevant keywords found. Normalize to lowercase.
            * `missing_suggestions` (list): List of important keywords/skills from the JD seemingly missing or underrepresented. Suggest up to 5 potentially relevant ones. Normalize to lowercase.
            * `density_impression` (string): Qualitative assessment of keyword usage (e.g., "Good density and relevance", "Fair, some keywords used", "Low density, key terms missing").
        3.  `clarity_structure_check` (string): Brief assessment of clarity and organization for parsing (e.g., "Clear structure, easy to parse", "Generally clear", "Lacks clear sections/paragraphs").
        4.  `formatting_check` (string): Brief assessment of ATS suitability regarding formatting (e.g., "Standard format", "Appears clean", "Check complex elements"). For Resumes, mention standard sections.
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
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)
            results = json.loads(raw_json)
            required_keys = ["ats_score", "keyword_check", "clarity_structure_check", "formatting_check", "overall_feedback"]
            if not all(key in results for key in required_keys):
                 raise ValueError("LLM validation response missing required keys.")
            kw_check = results.get("keyword_check")
            if not isinstance(kw_check, dict) or not all(k in kw_check for k in ["found_keywords", "missing_suggestions", "density_impression"]):
                 raise ValueError("LLM validation response keyword_check structure is invalid.")
             # Ensure lists are lists of strings
            kw_check["found_keywords"] = [str(kw).lower().strip() for kw in kw_check.get("found_keywords",[]) if str(kw).strip()]
            kw_check["missing_suggestions"] = [str(kw).lower().strip() for kw in kw_check.get("missing_suggestions",[]) if str(kw).strip()]


            logging.info(f"ATS Validation successful for {document_type}.")
            results["error"] = None # Explicitly set error to None on success

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