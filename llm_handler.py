# llm_handler.py

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse # For streaming type hint
import asyncio
import json
import re # For parsing and highlighting
import logging
# Make sure 'Union' is included in this line:
from typing import Optional, Dict, List, Tuple, Any, AsyncGenerator, Union

GENERATION_MODEL_NAME = 'gemini-1.5-flash'
EXTRACTION_MODEL_NAME = 'gemini-1.5-flash'

# Generation Types
TYPE_RESUME = "RESUME"
TYPE_COVER_LETTER = "COVER_LETTER"
TYPE_EMAIL = "EMAIL"
TYPE_LINKEDIN_MSG = "LINKEDIN_REFERRAL_MSG" # <-- NEW TYPE

# Email Recipient Types (Keep as is)
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
) -> Union[str, AsyncGenerator[str, None]]: # Ensure Union is imported from typing
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

            return stream_generator() # Return the async generator *function's result* (the generator object)
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

# llm_handler.py
# --- ENHANCED & STRICT LENGTH: LinkedIn Referral Message Generator (Async) ---
async def generate_linkedin_message(
    name: str,
    job_description_data: Dict,
    resume_data: Dict,
    google_api_key: str,
    tone: str,
    connection_name: Optional[str] = None
) -> Optional[str]:
    """Generates a concise LinkedIn message (295-300 chars) focused on referrals."""
    try:
        await _configure_gemini(google_api_key)

        # Extract context (same as before)
        jd_title = job_description_data.get('job_title', 'the open role')
        jd_company = job_description_data.get('company_name', 'your company')
        company_context = job_description_data.get('company_values_mission_challenges', '')
        top_req = resume_data.get("ranked_jd_requirements", [None])[0]
        resume_keywords = resume_data.get("extracted_keywords", [])
        key_highlight = top_req if top_req else (resume_keywords[0] if resume_keywords else "my relevant background")
        connection_reason = ""
        if connection_name: connection_reason = f"I saw your profile/activity related to [Specific Area] at {jd_company} and was impressed."
        else: connection_reason = f"I'm keenly interested in {jd_company}'s work{' in ' + company_context if company_context else ''}."

        # Define the ENHANCED & STRICT LENGTH prompt
        prompt = f"""
        **Role:** Expert LinkedIn Networking Strategist crafting concise, high-impact connection requests.
        **Goal:** Generate a personalized, professional, compelling LinkedIn connection request message for `{name}`. The message MUST grab attention and motivate the recipient to respond positively regarding the `{jd_title}` role at `{jd_company}`.

        **Candidate Name:** {name}
        **Target Job Title:** {jd_title}
        **Target Company:** {jd_company}
        **Company Context:** {company_context}
        **Candidate's Key Relevance:** Aligns with `{key_highlight}` requirement/skill.
        **Intended Connection Name (Optional):** {connection_name or 'Unknown'}
        **Desired Tone:** {tone} (professional, respectful, initiative)

        **Instructions:**
        1.  **STRICT Character Limit:** The FINAL message MUST be **under 300 characters**. Aim for the **290-299 character range**. Be extremely concise. Use abbreviations (e.g., "exp.") if necessary ONLY if it maintains clarity.
        2.  **Compelling Hook:** Start with a brief, specific, genuine point of connection (use/adapt `connection_reason` or create plausible alternative).
        3.  **Show Relevance Quickly:** State interest in the `{jd_title}` role and concisely link `Candidate's Key Relevance` (e.g., "My exp. in {key_highlight} seems relevant...").
        4.  **Strategic Ask:** Politely ask for brief insight OR if they could suggest the right contact for potential referrals.
        5.  **Tone:** Professional, respectful, concise, proactive, '{tone}'.
        6.  **AVOID Generic Phrases:** No "My background aligns well", "Seeking insights/referral", "Hope you are well".
        7.  **Output ONLY the message text.** No "Hi [Name]," prefix, no subject, no signature.

        **Connection Reason Example (Adapt/Use):** {connection_reason}
        """

        logging.info("Generating LinkedIn Referral Message (Strict Length)...")
        message = await _call_llm_async(prompt, google_api_key, GENERATION_MODEL_NAME, temperature=0.55) # Slightly lower temp

        # --- Strict Post-Processing Length Check & Truncation ---
        if message:
            max_len = 300
            if len(message) > max_len:
                original_len = len(message)
                logging.warning(f"Generated LinkedIn message > {max_len} chars ({original_len}). Force truncating.")
                # Find last space or sentence punctuation strictly *before* max_len
                cut_off_point = -1
                for char in ['.', '?', '!', ' ']:
                    # Search up to one character before the max length
                    point = message[:max_len].rfind(char)
                    # Prioritize breaks closer to the end, but after a reasonable minimum length
                    if point > max(cut_off_point, max_len - 50): # Try to break after char 250
                        cut_off_point = point

                if cut_off_point != -1:
                     # Truncate and add ellipsis if not ending on punctuation
                     message = message[:cut_off_point+1]
                     if message[-1] not in ['.','?','!']: message += "..."
                     logging.info(f"Cleanly truncated message to {len(message)} chars.")
                else: # Hard truncate if no good break point found
                     message = message[:max_len-3] + "..."
                     logging.info(f"Hard truncated message to {len(message)} chars.")
            else:
                 logging.info(f"Generated LinkedIn message length: {len(message)} chars.") # Log length if within limit


        return message

    except Exception as e:
        logging.error(f"Error generating LinkedIn message: {e}", exc_info=True)
        return f"Error generating message. Details: {str(e)[:100]}"
    


# --- Enhanced Resume Parsing (Sections, Keywords, Actions/Results) ---
async def parse_resume_advanced_llm(resume_content: str, google_api_key: str) -> Dict[str, Any]:
    """
    Parses resume into sections, extracts keywords, and identifies action verbs + quantifiable results.
    Returns dict containing 'sections', 'extracted_keywords', 'achievements', and 'error' (None on success).
    """
    prompt = f"""
    Analyze the following resume text. Perform three tasks:
    1. Parse into logical sections (Summary, Skills, Experience, Education, Projects, etc.). Use standardized keys. Preserve original text.
    2. Extract a comprehensive list of specific skills, technologies, tools, and methodologies mentioned (lowercase, unique).
    3. From the 'Experience' section ONLY, extract key achievements as a list of objects, each containing 'action_verb' (string, the leading verb) and 'quantifiable_result' (string, the part describing a number/metric/scale, or null if none).

    Return ONLY a valid JSON object with three top-level keys:
    - "sections": Object containing parsed section texts.
    - "extracted_keywords": List of unique skill/keyword strings (lowercase).
    - "achievements": List of objects, e.g., [{{"action_verb": "Managed", "quantifiable_result": "budget of $5M"}}, {{"action_verb": "Increased", "quantifiable_result": "efficiency by 15%"}}, {{"action_verb": "Developed", "quantifiable_result": null}}].

    Resume Text:
    ---
    {resume_content}
    ---

    JSON Output:
    """
    # Default structure in case of errors
    error_result = {"sections": {"full_text": resume_content}, "extracted_keywords": [], "achievements": [], "error": "Initialization error"}
    try:
        response_text = await _call_llm_async(prompt, google_api_key, EXTRACTION_MODEL_NAME, 0.1, True)
        # Robust JSON parsing
        try:
            # Initial cleanup
            raw_json = response_text.strip().replace('```json', '').replace('```', '').strip()
            # Handle potential trailing commas
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)

            # *** ADD CONTROL CHARACTER CLEANING ***
            # Remove characters in the C0 control character range (U+0000 to U+001F)
            # except for tab (\t), newline (\n), carriage return (\r), form feed (\f), backspace (\b)
            # Using regex to replace unwanted control chars with empty string
            control_chars_pattern = r'[\x00-\x08\x0b\x0c\x0e-\x1f\x80-\x9f]' # <-- EXPANDED RANGE
            cleaned_raw_json = re.sub(control_chars_pattern, '', raw_json)
            # ********************************

            parsed_json = json.loads(cleaned_raw_json) # Try parsing the more thoroughly cleaned string

            # --- Validation (remains the same) ---
            if not isinstance(parsed_json, dict): raise ValueError("LLM did not return a dictionary.")
            sections = parsed_json.get("sections"); keywords_raw = parsed_json.get("extracted_keywords"); achievements_raw = parsed_json.get("achievements")
            if not isinstance(sections, dict): sections = {"full_text": resume_content}; logging.warning("Invalid 'sections' structure.")
            if not isinstance(keywords_raw, list): keywords = []; logging.warning("Invalid 'extracted_keywords' structure.")
            else: keywords = sorted(list(set([str(kw).lower().strip() for kw in keywords_raw if str(kw).strip()])))
            if not isinstance(achievements_raw, list): achievements = []; logging.warning("Invalid 'achievements' structure.")
            else: # Validate achievement structure
                achievements = []
                for item in achievements_raw:
                     if isinstance(item, dict) and 'action_verb' in item:
                         achievements.append({
                             "action_verb": str(item.get("action_verb")).strip(),
                             "quantifiable_result": str(item.get("quantifiable_result")).strip() if item.get("quantifiable_result") else None
                         })
                     else: logging.warning(f"Skipping invalid achievement item: {item}")

            logging.info("Resume sections, keywords, and achievements parsed successfully.")
            return {"sections": sections, "extracted_keywords": keywords, "achievements": achievements, "error": None}

        except json.JSONDecodeError as json_err:
            # *** ENHANCED LOGGING ***
            try:
                problem_char = cleaned_raw_json[json_err.pos]
                problem_char_code = ord(problem_char)
                logging.error(
                    f"Failed to parse resume section/keyword JSON: {json_err}. "
                    f"Problematic character: '{problem_char}' (Code: {problem_char_code}, Hex: {hex(problem_char_code)}) at pos {json_err.pos}. "
                    f"Cleaned JSON Snippet (around error): '{cleaned_raw_json[max(0, json_err.pos-30):json_err.pos+30]}' "
                    f"Original Raw Snippet: '{raw_json[max(0, json_err.pos-30):json_err.pos+30]}...'"
                )
            except IndexError: # Error might be at the very end
                 logging.error(
                    f"Failed to parse resume section/keyword JSON: {json_err}. "
                    f"Error position {json_err.pos} might be out of bounds or related to overall structure. "
                    f"Cleaned JSON Snippet (end): '...{cleaned_raw_json[-60:]}' "
                    f"Original Raw Snippet (end): '...{raw_json[-60:]}'"
                 )
            # **************************
            error_result["error"] = f"Invalid JSON response for resume parsing: {json_err} (Pos: {json_err.pos}) even after extended cleaning." # Updated message
            error_result["raw_response_snippet"] = raw_json[max(0, json_err.pos-50):json_err.pos+50]
            return error_result

    except Exception as e:
        logging.error(f"Error parsing advanced resume with LLM: {e}", exc_info=True)
        error_result["error"] = f"Advanced Resume Parsing Error: {e}"
        return error_result

# --- Enhanced JD Extraction (including Company Context) ---
async def _extract_structured_data_enhanced_jd(text: str, google_api_key: str) -> Dict[str, Any]:
    """Extracts structured data from JD, including company context."""
    schema = {
        "job_title": "string", "company_name": "string",
        "key_skills_requirements": ["string (distinct skills/tools/experience levels)"],
        "location": "string", "summary": "string (1-2 sentence role objective)",
        "company_values_mission_challenges": "string (Brief notes on company culture, goals, or challenges mentioned)"
    }
    instructions = "Extract required fields. For 'key_skills_requirements', list distinct technical and soft skills, tools, platforms, and experience duration (e.g., '5+ years'). For 'company_values_mission_challenges', capture phrases related to company culture, goals, values, or specific problems they are trying to solve."
    prompt = f"""
    Analyze the following Job Description and extract information according to the schema.
    Return ONLY a valid JSON object. Use null if information is not found.
    Instructions: {instructions}
    Schema: {json.dumps(schema)}

    Text:
    ---
    {text}
    ---

    JSON Output:
    """
    try:
        response_text = await _call_llm_async(prompt, google_api_key, EXTRACTION_MODEL_NAME, 0.1, True)
        try:
             raw_json = response_text.strip().replace('```json', '').replace('```', '').strip()
             raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)
             extracted_data = json.loads(raw_json)
             if not isinstance(extracted_data, dict): raise ValueError("LLM did not return dict for JD.")
        except json.JSONDecodeError as json_err: raise ValueError(f"Invalid JSON response for JD: {json_err}")

        logging.info("Enhanced structured data extracted for JD.")
        # Normalize requirements list
        reqs = extracted_data.get("key_skills_requirements")
        if reqs is not None and not isinstance(reqs, list):
             if isinstance(reqs, str): extracted_data["key_skills_requirements"] = [item.strip() for item in re.split(r'[,\n;•]', reqs) if item.strip()]
             else: extracted_data["key_skills_requirements"] = []
        elif reqs is None: extracted_data["key_skills_requirements"] = []

        return {"data": extracted_data, "error": None}
    except Exception as e:
        logging.error(f"Error extracting enhanced JD data: {e}", exc_info=True)
        return {"data": {}, "error": f"Enhanced JD Extraction Error: {e}"}

# --- Rank JD Requirements (Async LLM) ---
async def _rank_jd_requirements(requirements: List[str], google_api_key: str) -> Dict[str, Any]:
    """Uses an LLM to rank JD requirements (async). Returns dict with data or error."""
    if not requirements: return {"ranked_list": [], "error": None}

    req_string = "\n".join(f"- {req.strip()}" for req in requirements if req.strip())
    if not req_string: return {"ranked_list": [], "error": None}

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
        response_text = await _call_llm_async(prompt, google_api_key, EXTRACTION_MODEL_NAME, 0.2, True)
        try:
            raw_json = response_text.strip().replace('```json', '').replace('```', '').strip()
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)
            ranked_data = json.loads(raw_json)
            if not isinstance(ranked_data.get("ranked_list"), list): raise ValueError("LLM invalid JSON list for 'ranked_list'.")
        except json.JSONDecodeError as json_err: raise ValueError(f"Invalid JSON response for JD ranking: {json_err}")

        logging.info("JD Requirements ranked successfully.")
        return {"ranked_list": ranked_data["ranked_list"], "error": None}
    except Exception as e:
        logging.error(f"Error ranking JD requirements: {e}", exc_info=True)
        return {"ranked_list": requirements, "error": f"Ranking Error: {e}"}


# --- Data Preparation (Async) ---
async def _prepare_common_data(job_description: str, resume_content: str, google_api_key: str) -> Dict[str, Any]:
    """Orchestrates enhanced data extraction, parsing, ranking, keyword comparison."""
    results = { # Initialize with defaults/fallback structures
        "error": None, "jd_data": {},
        "resume_sections": {"full_text": resume_content}, "resume_keywords": [], "resume_achievements": [],
        "ranked_jd_requirements": [], "missing_keywords_from_resume": []
    }
    task_errors = []
    try:
        # Concurrently run enhanced JD extraction and enhanced Resume parsing
        tasks = {
            "jd_extraction": _extract_structured_data_enhanced_jd(job_description, google_api_key),
            "resume_parsing_advanced": parse_resume_advanced_llm(resume_content, google_api_key),
        }
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        result_map = dict(zip(tasks.keys(), task_results))

        # Process JD Extraction results
        jd_res = result_map.get("jd_extraction")
        if isinstance(jd_res, Exception): task_errors.append(f"JD Extraction failed: {jd_res}")
        elif jd_res.get("error"): task_errors.append(f"JD Extraction error: {jd_res['error']}")
        else: results["jd_data"] = jd_res.get("data", {})

        # Process Resume Parsing results
        resume_res = result_map.get("resume_parsing_advanced")
        if isinstance(resume_res, Exception): task_errors.append(f"Resume Parsing failed: {resume_res}")
        elif resume_res.get("error"): task_errors.append(f"Resume Parsing error: {resume_res['error']}")
        else:
            results["resume_sections"] = resume_res.get("sections", {"full_text": resume_content})
            results["resume_keywords"] = resume_res.get("extracted_keywords", [])
            results["resume_achievements"] = resume_res.get("achievements", [])

        # Rank JD requirements if successfully extracted
        extracted_requirements = results.get("jd_data", {}).get("key_skills_requirements", [])
        if extracted_requirements and isinstance(extracted_requirements, list):
            # Run ranking sequentially after extraction needed for input
            ranking_res = await _rank_jd_requirements(extracted_requirements, google_api_key)
            if isinstance(ranking_res, Exception): task_errors.append(f"JD Ranking failed: {ranking_res}")
            elif ranking_res.get("error"): task_errors.append(f"JD Ranking error: {ranking_res['error']}")
            else: results["ranked_jd_requirements"] = ranking_res.get("ranked_list", [])
        elif results.get("jd_data"): results["ranked_jd_requirements"] = []
        else: results["ranked_jd_requirements"] = []

        # Identify Missing Keywords
        if results.get("jd_data") and "resume_keywords" in results:
             try:
                jd_keywords_lower = set(str(kw).lower().strip() for kw in results.get("jd_data", {}).get("key_skills_requirements", []) if str(kw).strip())
                resume_keywords_lower = set(results.get("resume_keywords", [])) # Already lower
                missing_keywords = sorted(list(jd_keywords_lower - resume_keywords_lower))
                results["missing_keywords_from_resume"] = missing_keywords
                logging.info(f"Identified potentially missing keywords: {missing_keywords}")
             except Exception as kw_err: task_errors.append(f"Keyword comparison error: {kw_err}"); results["missing_keywords_from_resume"] = []
        else: results["missing_keywords_from_resume"] = []; logging.info("Skipping missing keyword identification.")

        if task_errors: results["error"] = "; ".join(task_errors)
        results["raw_resume"] = resume_content

    except Exception as e:
        results["error"] = f"Unexpected Error in Data Preparation: {e}"
        logging.error(f"Critical error in _prepare_common_data: {e}", exc_info=True)

    return results


# --- Generator with Multi-Turn Refinement & Enhanced Instructions (Async Stream) ---
async def generate_application_text_streamed(
    name: str, email: str, job_description: str, resume_content: str,
    generation_type: str, google_api_key: str, tone: str
) -> AsyncGenerator[str, None]:
    """Generates Resume or Cover Letter using refinement, enhanced prompts (async stream)."""
    common_data = {}
    try:
        # --- 1. Prepare Data ---
        common_data = await _prepare_common_data(job_description, resume_content, google_api_key)
        if common_data.get("error"): yield f"\n--- ERROR prep: {common_data['error']} ---\n"; return
        if not common_data.get("jd_data") or not common_data.get("resume_sections"): yield "\n--- ERROR: Missing JD/Resume data ---\n"; return

        # --- Format inputs ---
        resume_sections_str = "\n\n".join(f"**{sec.upper()}**\n{content}" for sec, content in common_data.get("resume_sections", {}).items() if content)
        MAX_RESUME_PROMPT_LEN = 6000
        if len(resume_sections_str) > MAX_RESUME_PROMPT_LEN: resume_sections_str = resume_sections_str[:MAX_RESUME_PROMPT_LEN] + "\n... [Resume Truncated] ..."

        ranked_req_str = "\n".join(f"- {req}" for req in common_data.get("ranked_jd_requirements", [])) or "N/A"
        jd_title = common_data.get('jd_data', {}).get('job_title', 'N/A')
        jd_company = common_data.get('jd_data', {}).get('company_name', 'N/A')
        company_context = common_data.get('jd_data', {}).get('company_values_mission_challenges', 'N/A')
        missing_keywords = common_data.get("missing_keywords_from_resume", [])
        missing_keywords_str = ", ".join(f"`{kw}`" for kw in missing_keywords) if missing_keywords else "None identified"
        extracted_achievements = common_data.get("resume_achievements", [])

        # --- 2. Initial Draft Prompt (Enhanced) ---
        draft_prompt = f"""
        **Candidate:** {name} ({email})
        **Target Job:** {jd_title} at {jd_company}
        **Company Context:** {company_context}
        **Ranked Requirements:**\n{ranked_req_str}
        **Parsed Resume Sections:**\n{resume_sections_str}
        **Extracted Resume Achievements:** {json.dumps(extracted_achievements[:10])} [Sample]
        **Desired Tone:** {tone}
        **Potentially Missing Keywords:** {missing_keywords_str}

        **Task:** Generate **FIRST DRAFT** of a {generation_type} for ATS and humans.
        1. Address top ranked requirements with strong evidence from resume sections/achievements.
        2. Use strong action verbs & quantifiable results (leverage extracted achievements).
        3. Incorporate 'Potentially Missing Keywords' naturally where supported by experience. **Do NOT invent skills.** Flag uncertainty: `[Note: Assumed relevance based on X]`.
        4. {'Align with Company Context.' if generation_type != TYPE_RESUME else ''}
        5. Adhere to '{tone}'.
        **Format:** {'ATS-friendly Markdown (## Headings, bullets). Prioritize Summary, Skills, Experience, Education.' if generation_type == TYPE_RESUME else 'Standard Cover Letter (Intro, Body, Conclusion).'}
        {'For COVER LETTER: Aim for approx 300-450 words.' if generation_type == TYPE_COVER_LETTER else ''}

        **Output ONLY the {generation_type} draft:**
        """

        # --- 3. Generate Initial Draft ---
        yield f"--- Generating initial {generation_type} draft... ---\n"
        initial_draft = await _call_llm_async(draft_prompt, google_api_key, GENERATION_MODEL_NAME, 0.6)
        if not initial_draft: raise ValueError("Initial draft generation failed.")

        # --- 4. Critique Prompt (Enhanced) ---
        critique_prompt = f"""
        **Target Job Requirements (Ranked):**\n{ranked_req_str}
        **Potentially Missing Keywords Attempted:** {missing_keywords_str}
        **Initial {generation_type} Draft:**
        ---
        {initial_draft}
        ---

        **Task:** Critique the draft based ONLY on these criteria:
        1. **Requirement Alignment/Evidence:** Addresses top requirements with specific, quantified evidence?
        2. **Keyword Integration:** Are JD keywords (incl. 'missing' ones) used naturally? Any awkwardness? Are `[Note: ...]` flags used appropriately?
        3. **Action Verbs/Quantification:** Strong verbs? Quantified results present and effective?
        4. **Clarity/Conciseness/Tone:** Readability, length (for CL), tone ('{tone}')?
        5. **ATS Friendliness:** Structure, keyword usage.

        **Output ONLY the critique points (brief bullets):**
        """

        # --- 5. Generate Critique ---
        yield f"\n--- Generating critique... ---\n"
        critique = await _call_llm_async(critique_prompt, google_api_key, EXTRACTION_MODEL_NAME, 0.3)
        if not critique: logging.warning("Critique empty."); critique = "No critique generated."


        # --- 6. Refinement Prompt (Enhanced) ---
        refinement_prompt = f"""
        **Candidate:** {name} ({email})
        **Target Job:** {jd_title} at {jd_company}
        **Company Context:** {company_context}
        **Ranked Requirements:**\n{ranked_req_str}
        **Parsed Resume Sections:**\n{resume_sections_str}
        **Extracted Resume Achievements:** {json.dumps(extracted_achievements[:10])} [Sample]
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

        **Task:** Generate the **FINAL, REVISED** {generation_type} by meticulously addressing the critique.
        * Enhance requirement alignment using strong, specific, quantified evidence.
        * Improve natural integration of relevant JD keywords (esp. 'missing' ones). **Remove/justify `[Note: ...]` flags.** Optimize action verbs.
        * Ensure clarity, conciseness, '{tone}', and ATS-friendly structure/formatting.
        {'For COVER LETTER: Maintain approx 300-450 words.' if generation_type == TYPE_COVER_LETTER else ''}
        {'For RESUME: Ensure valid ATS-friendly Markdown with standard sections (## Summary, ## Skills, ## Experience using bullet points, ## Education). Use action verbs and quantify.' if generation_type == TYPE_RESUME else ''}
        **Output ONLY the final {generation_type}. No extra commentary.**
        """

        # --- 7. Generate Final Version (Async Stream) ---
        yield f"\n--- Generating final refined {generation_type}... ---\n"
        # *** CORRECTED PART ***
        # Await the call to get the async generator object first
        final_stream_generator = await _call_llm_async(
            refinement_prompt, google_api_key, GENERATION_MODEL_NAME,
            temperature=0.5, stream=True
        )

        # Check if we got an async generator before iterating
        if hasattr(final_stream_generator, '__aiter__'):
            try:
                async for chunk in final_stream_generator:
                     yield chunk
            except Exception as stream_iter_err:
                error_message = f"\n--- Error iterating through final stream: {stream_iter_err} ---"
                logging.error(error_message, exc_info=True)
                yield error_message
        # Handle cases where _call_llm_async might not return a generator (e.g., error before streaming start)
        elif isinstance(final_stream_generator, str):
             error_message = f"\n--- ERROR: Expected stream generator, received string instead: {final_stream_generator[:100]}... ---"
             logging.error(error_message)
             yield error_message
        else:
             error_message = f"\n--- ERROR: Unexpected return type ({type(final_stream_generator)}) for final stream ---"
             logging.error(error_message)
             yield error_message
        # *** END CORRECTION ***

    except Exception as e:
        error_message = f"\n--- Error during {generation_type} Generation: {e} ---"
        logging.error(f"Error in generate_application_text_streamed: {e}", exc_info=True)
        yield error_message
        if common_data: yield f"\nDebug Info (Prep Error): {common_data.get('error') or 'OK'}"


# --- Email Generator + Validator (Async) ---
async def generate_email_and_validate(
    name: str, email: str, job_description: str, resume_content: str,
    google_api_key: str, tone: str, email_recipient_type: str,
    job_link: Optional[str] = None # Existing argument
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Generates a more impactful email, incorporating job link, and validates (async)."""
    generated_email = None
    validation_results = None
    common_data = {}
    try:
        # --- 1. Prepare Data ---
        common_data = await _prepare_common_data(job_description, resume_content, google_api_key)
        if common_data.get("error"): return None, {"error": f"Prep Error: {common_data['error']}"}
        if not common_data.get("jd_data"): return None, {"error": "Missing JD data."}

        # --- Format inputs ---
        resume_sections_str = "\n\n".join(f"**{sec.upper()}**\n{content}" for sec, content in common_data.get("resume_sections", {}).items() if content)
        MAX_RESUME_PROMPT_LEN = 4000 # Truncate if needed
        if len(resume_sections_str) > MAX_RESUME_PROMPT_LEN: resume_sections_str = resume_sections_str[:MAX_RESUME_PROMPT_LEN] + "\n... [Resume Truncated] ..."

        ranked_req_str = "\n".join(f"- {req}" for req in common_data.get("ranked_jd_requirements", [])) or "N/A"
        jd_title = common_data.get('jd_data', {}).get('job_title', 'N/A')
        jd_company = common_data.get('jd_data', {}).get('company_name', 'N/A')
        company_context = common_data.get('jd_data', {}).get('company_values_mission_challenges', 'N/A')
        missing_keywords = common_data.get("missing_keywords_from_resume", [])
        missing_keywords_str = ", ".join(f"`{kw}`" for kw in missing_keywords) if missing_keywords else "None identified"
        extracted_achievements = common_data.get("resume_achievements", [])

        # Define recipient focus (remains same)
        recipient_focus = ""
        if email_recipient_type == RECIPIENT_TA_HM: recipient_focus = "Focus on results, ROI, problem-solving, direct experience match, value proposition."
        else: recipient_focus = "Focus on transferable skills, clear communication, enthusiasm, professionalism."

        # Prepare job link display for prompt context
        job_link_display = job_link if job_link else "Not Provided"

        # --- 2. Generate Email (Async - UPDATED Prompt) ---
        email_prompt = f"""
        **Role:** Expert Email Copywriter for impactful, ATS-friendly job applications.
        **Goal:** Generate a compelling email (approx. 200-275 words) for `{name}` to make the recipient prioritize the application for the `{jd_title}` role at `{jd_company}`.

        **Candidate Info:** {name} ({email})
        **Target Job:** {jd_title} at {jd_company}
        **Job Posting Link:** {job_link_display}  {'''<-- Link provided''' if job_link else ''}  #<-- JOB LINK CONTEXT
        **Company Context:** {company_context}
        **Ranked Job Requirements:**\n{ranked_req_str}
        **Parsed Candidate Resume Sections:**\n{resume_sections_str}
        **Extracted Resume Achievements:** {json.dumps(extracted_achievements[:5], indent=2)} [Sample]
        **Potentially Missing Keywords:** {missing_keywords_str}
        **Desired Tone:** {tone} (apply professionally, add confidence & impact) #<-- TONE
        **Recipient Type:** {email_recipient_type} ({recipient_focus}) #<-- RECIPIENT FOCUS

        **Instructions:**
        1.  **Subject Line:** Clear, professional (Job Title, Name).
        2.  **Hook/Connection:** Start strong... link candidate's core expertise to the top job need. #<-- STRONG OPENING
        3.  **Include Job Link (If Provided):** If '{job_link_display}' is a URL... incorporate it naturally... Omit if no link provided. #<-- JOB LINK INSTRUCTION
        4.  **Evidence-Based Body:** Showcase the single most impactful achievement/experience addressing a top ranked requirement. Quantify results (%, $, #) using resume data. Explain the IMPACT. {recipient_focus} #<-- IMPACTFUL EVIDENCE + QUANTIFICATION + RECIPIENT FOCUS
        5.  **Keyword Integration:** Naturally weave in 1-2 other relevant JD keywords (incl. 'Potentially Missing Keywords' if supported by experience). No inventing skills. Flag uncertainty: `[Note: ...]`. #<-- KEYWORD INCORPORATION
        6.  **Company Alignment:** Briefly link skills/goals to 'Company Context'. Show specific interest. #<-- COMPANY ALIGNMENT
        7.  **Value Proposition:** Clearly state the specific value `{name}` brings to *this* role/company. #<-- VALUE PROP
        8.  **Conciseness & Tone:** Maintain '{tone}'. Prioritize impact (approx. 200-275 words). Ensure ATS-friendly language. #<-- WORD COUNT + TONE + ATS
        9.  **Call to Action:** Conclude professionally, expressing strong enthusiasm for next steps & mentioning attached resume. #<-- CONFIDENT CTA

        **Output ONLY the email content (Subject line and Body):**
        Subject: [Your Subject Line Here]

        [Body of the email starts here]
        """
        logging.info("Generating Email with enhanced prompt (incl. job link)...")
        generated_email = await _call_llm_async(email_prompt, google_api_key, GENERATION_MODEL_NAME, temperature=0.7)
        if not generated_email: raise ValueError("Email generation failed or returned empty.")

        # --- 3. Validate Email (Async - remains the same) ---
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



# --- ATS Validator (Async - Enhanced Checks) ---
async def _validate_ats_friendliness(
    document_text: str, document_type: str, job_description_data: Dict, google_api_key: str
) -> Dict[str, Any]:
    """Uses an LLM to evaluate ATS friendliness with more granular checks (async)."""
    results = {"error": None}
    if not document_text: return {"error": "No document text provided."}
    try:
        jd_keywords_list = job_description_data.get("key_skills_requirements", [])
        jd_keywords_str = ", ".join(jd_keywords_list) if jd_keywords_list else "N/A"
        jd_title = job_description_data.get("job_title", "N/A")

        # Define detailed checks based on type
        detailed_checks_instructions = "* (No specific checks for this type)"
        if document_type == TYPE_RESUME:
            detailed_checks_instructions = """
            * `Standard Sections Present`: Check if standard headings (Summary, Skills, Experience, Education) seem present. (e.g., "Yes", "Partially", "No")
            * `Clear Date Formats`: Assess if experience/education dates appear standard/parsable. (e.g., "Good", "Fair", "Inconsistent")
            * `Action Verbs Used`: Does Experience section use action verbs effectively? (e.g., "Strong", "Moderate", "Limited")
            * `Quantifiable Results Present`: Are there indicators of quantified results? (e.g., "Yes", "Some", "Few/None")
            * `Contact Info Clear`: Is contact info likely present and clear? (e.g., "Yes", "Likely Missing")
            """
        elif document_type == TYPE_COVER_LETTER:
             detailed_checks_instructions = """
            * `Standard CL Structure`: Follows Intro, Body, Conclusion? (e.g., "Yes", "Somewhat", "No")
            * `Action Verbs Used`: Body uses action verbs? (e.g., "Good", "Limited")
            * `Quantifiable Results Present`: Metrics mentioned? (e.g., "Yes", "Few/None")
            """
        elif document_type == TYPE_EMAIL:
             detailed_checks_instructions = """
            * `Conciseness Check`: Is body appropriately brief? (e.g., "Good", "Long", "Brief")
            * `Clarity of Purpose`: Is the main point clear? (e.g., "Clear", "Unclear")
            * `Contact Info Present`: Signature includes contact info? (e.g., "Yes", "Missing")
            """

        prompt = f"""
        **Task:** Perform a detailed ATS friendliness evaluation of the '{document_type}' for job '{jd_title}'.
        **Job Description Keywords/Requirements:** {jd_keywords_str}

        **Document Text to Evaluate:**
        ---
        {document_text}
        ---

        **Evaluation Criteria & Output Format:**
        Return ONLY a valid JSON object with:
        1.  `ats_score` (integer 1-5): Overall ATS compatibility score.
        2.  `keyword_check` (object): {{ "found_keywords": [list], "missing_suggestions": [list], "density_impression": "string" }} (lowercase keywords).
        3.  `clarity_structure_check` (string): Assessment of clarity/organization for parsing.
        4.  `formatting_check` (string): Assessment of ATS-friendly formatting.
        5.  **`detailed_checks` (object): Provide brief assessments (e.g., "Yes", "No", "Good", "Needs Improvement", "N/A") for these points:**
            {detailed_checks_instructions}
        6.  `overall_feedback` (string): Brief, actionable feedback to improve ATS compatibility.
        """
        response_text = await _call_llm_async(prompt, google_api_key, EXTRACTION_MODEL_NAME, 0.2, True)

        # Robust JSON parsing and validation
        try:
            raw_json = response_text.strip().replace('```json', '').replace('```', '').strip()
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)
            results = json.loads(raw_json)
            required_keys = ["ats_score", "keyword_check", "clarity_structure_check", "formatting_check", "detailed_checks", "overall_feedback"]
            if not all(key in results for key in required_keys): raise ValueError("Validation response missing keys.")
            kw_check = results.get("keyword_check"); dt_check = results.get("detailed_checks")
            if not isinstance(kw_check, dict) or not all(k in kw_check for k in ["found_keywords", "missing_suggestions", "density_impression"]): raise ValueError("keyword_check structure invalid.")
            if not isinstance(dt_check, dict): raise ValueError("detailed_checks structure invalid.")
            # Sanitize keyword lists
            kw_check["found_keywords"] = sorted(list(set([str(kw).lower().strip() for kw in kw_check.get("found_keywords",[]) if str(kw).strip()])))
            kw_check["missing_suggestions"] = sorted(list(set([str(kw).lower().strip() for kw in kw_check.get("missing_suggestions",[]) if str(kw).strip()])))

            logging.info(f"Enhanced ATS Validation successful for {document_type}.")
            results["error"] = None

        except (json.JSONDecodeError, ValueError) as json_err:
             error_msg = f"Failed to parse/validate LLM validation response: {json_err}"
             logging.error(f"{error_msg}. Raw: {response_text[:500]}...")
             results = {"error": error_msg, "raw_response": response_text}
        except Exception as parse_err:
             error_msg = f"Error processing LLM validation response: {parse_err}"
             logging.error(error_msg, exc_info=True)
             results = {"error": error_msg, "raw_response": response_text}

    except Exception as e:
        logging.error(f"Error during ATS validation LLM call for {document_type}: {e}", exc_info=True)
        results = {"error": f"ATS Validation LLM Call Error: {e}"}

    return results