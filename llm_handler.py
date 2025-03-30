# llm_handler.py

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse # For streaming type hint
import asyncio
import json
import re # For parsing and highlighting
import logging
# Make sure 'Union' is included in this line:
from typing import Optional, Dict, List, Tuple, Any, AsyncGenerator, Union

GENERATION_MODEL_NAME = 'gemini-2.5-pro-exp-03-25' # Or your preferred generation model
EXTRACTION_MODEL_NAME = 'gemini-1.5-pro-latest' # Or your preferred extraction model

# Generation Types
TYPE_RESUME = "RESUME"
TYPE_COVER_LETTER = "COVER_LETTER"
TYPE_EMAIL = "EMAIL"
TYPE_LINKEDIN_MSG = "LINKEDIN_REFERRAL_MSG"

# Email Recipient Types
RECIPIENT_TA_HM = "Talent Acquisition / Hiring Manager"
RECIPIENT_GENERAL = "General Application / Unspecified"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [LLM_Handler] %(message)s')


# --- Helper: Robust LLM Call (Async) ---
async def _call_llm_async(
    prompt: str,
    api_key: str, # Keep api_key argument for potential direct use if needed
    model_name: str,
    temperature: float = 0.5,
    request_json: bool = False,
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]: # Ensure Union/AsyncGenerator are imported
    """General purpose async LLM caller with error handling and safety defaults."""
    gen_config = genai.GenerationConfig(temperature=temperature)
    # Default safety settings - blocking potentially harmful content
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    if request_json:
        # Attempt to set response MIME type for models supporting it (like Gemini 1.5 Pro)
        # Check your specific model documentation if needed.
        if hasattr(gen_config, 'response_mime_type') and '1.5' in model_name: # Simple check for 1.5
             try:
                  gen_config.response_mime_type = "application/json"
                  logging.info(f"Requesting JSON output via mime type for model '{model_name}'.")
             except Exception as e_mime:
                  # Log if setting mime type fails, but continue relying on prompt
                  logging.warning(f"Failed to set JSON mime type for model '{model_name}': {e_mime}. Relying on prompt instructions.")
        else:
            logging.warning(f"Model config for '{model_name}' may not support direct JSON mime type setting or is not 1.5+. Relying on prompt instructions for JSON format.")

    try:
        # NOTE: Ensure genai.configure(api_key=...) is called appropriately elsewhere
        # in your application startup, or pass the api_key directly if needed by
        # specific SDK methods (though usually configure handles it globally).

        model = genai.GenerativeModel(
            model_name,
            generation_config=gen_config,
            safety_settings=safety_settings
            # Pass api_key here IF the specific model/SDK version requires it per-call
            # and genai.configure isn't sufficient. Check SDK docs.
            # Example: api_key=api_key (if needed)
        )
        logging.info(f"Calling model '{model_name}' (stream={stream}, json={request_json})...")

        if stream:
            # Define and return the async generator for streaming
            async def stream_generator():
                try:
                    # Use await with generate_content_async for async streaming
                    response_stream = await model.generate_content_async(prompt, stream=True)
                    async for chunk in response_stream:
                        try:
                            # Check for text and potential blocking in the chunk
                            # Prioritize checking feedback before accessing potentially blocked text
                            if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                                block_reason_str = str(chunk.prompt_feedback.block_reason) # Get string representation
                                logging.warning(f"Stream chunk blocked by safety filter: {block_reason_str}")
                                yield f"\n--- ERROR: Content blocked by safety filter ({block_reason_str}) ---\n"
                                break # Stop streaming if blocked

                            if hasattr(chunk, 'text') and chunk.text:
                                yield chunk.text

                        except ValueError as val_err: # Often indicates blocked content access attempt
                             # Check prompt feedback again in case of ValueError accessing text
                            block_reason_str = "Unknown"
                            if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                                 block_reason_str = str(chunk.prompt_feedback.block_reason)
                            logging.warning(f"Stream chunk ValueError (likely safety block: {block_reason_str}): {val_err}")
                            yield f"\n--- ERROR: Content blocked by safety filter ({block_reason_str}) ---\n"
                            break # Stop streaming
                        except Exception as e_chunk:
                            logging.error(f"Error processing stream chunk: {e_chunk}", exc_info=True)
                            yield f"\n--- ERROR Processing Stream Chunk: {e_chunk} ---\n"
                            break # Stop on chunk processing error
                except Exception as stream_err:
                    logging.error(f"Error initiating or processing stream for model '{model_name}': {stream_err}", exc_info=True)
                    yield f"\n--- ERROR DURING STREAMING: {stream_err} ---\n"

            return stream_generator() # Return the async generator *function's result* (the generator object)
        else:
            # Non-streaming call
            response = await model.generate_content_async(prompt)
            logging.info(f"Model '{model_name}' finished.")

            # Centralized safety check for non-streaming response
            if not response.candidates:
                block_reason = "Unknown"
                finish_reason_val = "Unknown" # Using different var name
                if response.prompt_feedback:
                     # Use getattr for safer access to block_reason enum/attribute
                     block_reason_attr = getattr(response.prompt_feedback, 'block_reason', None)
                     block_reason = str(block_reason_attr) if block_reason_attr else "Not Blocked"

                # Candidate finish reason might also be relevant, check candidate existence
                if response.candidates and response.candidates[0]:
                     finish_reason_val = getattr(response.candidates[0], 'finish_reason', "N/A")

                logging.error(f"Response blocked or empty. Block Reason: {block_reason}. Finish Reason: {finish_reason_val}. Prompt Feedback: {response.prompt_feedback}")
                # Raise a specific error to indicate failure
                raise ValueError(f"Response blocked by safety filter (Reason: {block_reason}) or no candidates generated (Finish Reason: {finish_reason_val}).")

            # Access text safely only if candidates exist and are not blocked
            try:
                # Check candidate finish reason before accessing text
                candidate = response.candidates[0]
                finish_reason_val = getattr(candidate, 'finish_reason', "UNKNOWN")
                if str(finish_reason_val) != "STOP": # Check if generation finished normally
                     logging.warning(f"Candidate generation finished with reason: {finish_reason_val}. Text might be incomplete.")
                     # Optionally raise error or just proceed if partial text is acceptable

                # Access text via parts if direct .text access fails or for more control
                response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()

                # Clean JSON if requested and it seems like JSON markdown
                if request_json:
                    # More robust cleaning: remove ```json ... ``` markdown blocks first
                    match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        response_text = match.group(1).strip()
                        # Basic check if the extracted part looks like JSON
                        if not (response_text.startswith(('{', '[')) and response_text.endswith(('}', ']'))):
                              logging.warning("Extracted content from ```json block doesn't look like JSON.")
                              # Decide fallback: Use original or raise error? Let's log and use it for now.
                    else:
                         # Fallback if no ```json block found: Remove markers if they exist
                         response_text = re.sub(r"^```json\s*|\s*```$", "", response_text, flags=re.MULTILINE).strip()
                         # Basic check again
                         if not (response_text.startswith(('{', '[')) and response_text.endswith(('}', ']'))):
                              logging.warning("JSON requested, but response doesn't look like JSON after cleaning markers.")

                return response_text

            except ValueError as val_err: # This specific error often means blocked content access
                # Check feedback again, as the block might occur here
                block_reason = "Unknown"
                if response.prompt_feedback:
                     block_reason_attr = getattr(response.prompt_feedback, 'block_reason', None)
                     block_reason = str(block_reason_attr) if block_reason_attr else "Not Blocked"
                logging.error(f"Response text access error (likely safety block: {block_reason}): {val_err}. Prompt Feedback: {response.prompt_feedback}", exc_info=True)
                raise ValueError(f"Response blocked by safety filter (access error: {block_reason}): {val_err}")
            except IndexError:
                 # Handle case where response.candidates is empty after the initial check (shouldn't happen ideally)
                 logging.error("Response has no candidates even after initial check passed.")
                 raise RuntimeError("Failed to access response candidate.")
            except Exception as e_text:
                logging.error(f"Error extracting text from response parts: {e_text}", exc_info=True)
                raise RuntimeError(f"Failed to extract text from response: {e_text}")

    except Exception as e:
        logging.error(f"LLM call to '{model_name}' failed: {e}", exc_info=True)
        # Re-raise a more informative error or handle as needed
        raise RuntimeError(f"LLM API call failed for model '{model_name}': {e}")


# --- ENHANCED & STRICT LENGTH: LinkedIn Referral Message Generator (Async) ---
async def generate_linkedin_message(
    name: str,
    job_description_data: Dict,
    resume_data: Dict,
    google_api_key: str,
    tone: str,
    connection_name: Optional[str] = None
) -> Optional[str]:
    """Generates a concise LinkedIn message (aiming for <300 chars) focused on referrals."""
    try:
        # --- Context Extraction ---
        jd_title = job_description_data.get('job_title', 'the open role')
        jd_company = job_description_data.get('company_name', 'your company')
        company_context = job_description_data.get('company_values_mission_challenges', '')

        # Determine the key relevance point: Top ranked JD req > First resume keyword > Generic fallback
        top_req = resume_data.get("ranked_jd_requirements", [None])[0] # Get first item safely
        resume_keywords = resume_data.get("extracted_keywords", [])
        key_highlight = "my relevant background" # Default fallback
        if top_req:
            key_highlight = top_req
        elif resume_keywords:
            key_highlight = resume_keywords[0]

        # Truncate key_highlight if it's excessively long for the message context
        if len(key_highlight) > 60: key_highlight = key_highlight[:57] + "..."

        # Craft a concise connection reason
        connection_reason = ""
        if connection_name:
             # More concise and slightly personalized if name is known
             connection_reason = f"Impressed by your profile/activity related to {jd_company}."
        else:
             # Focus on company interest, use context if concise
             if company_context and len(company_context) < 80: # Check length before including
                  connection_reason = f"Keenly interested in {jd_company}'s work, particularly regarding {company_context}."
             else:
                  connection_reason = f"Keenly interested in {jd_company}'s work and the {jd_title} role."


        # --- Prompt Definition ---
        prompt = f"""
        **Role:** Expert LinkedIn Networking Strategist crafting concise, high-impact connection requests.
        **Goal:** Generate a personalized, professional, compelling LinkedIn connection request message for `{name}` to send to '{connection_name or 'a relevant contact'}'. The message MUST grab attention and motivate the recipient to consider responding positively regarding the `{jd_title}` role at `{jd_company}`.

        **Candidate Name:** {name}
        **Target Job Title:** {jd_title}
        **Target Company:** {jd_company}
        **Candidate's Key Relevance:** Aligns with `{key_highlight}` requirement/skill.
        **Intended Connection Name (Optional):** {connection_name or 'Unknown'}
        **Desired Tone:** {tone} (professional, respectful, concise, proactive)

        **Instructions:**
        1.  **STRICT Character Limit:** The FINAL message MUST be **under 300 characters**. Aim for the **290-299 character range**. Be extremely concise. Use standard abbreviations (e.g., "exp.") only if necessary for brevity and clarity is maintained.
        2.  **Compelling Hook:** Start with a brief, specific point of connection. Use/adapt the example: '{connection_reason}'. If no specific connection point, focus directly on interest in the company/role.
        3.  **Show Relevance Quickly:** State interest in the `{jd_title}` role and concisely link the candidate's relevance (e.g., "My exp. in '{key_highlight}' seems particularly relevant...").
        4.  **Strategic Ask:** Politely ask for *either* brief insight *or* if they might know the appropriate contact for referral consideration. Avoid demanding language. Example phrasing: "Would you be open to a brief chat, or perhaps could point me to the right contact?" or "Any insight you could share would be greatly appreciated."
        5.  **Tone:** Adhere strictly to: Professional, respectful, concise, proactive, '{tone}'.
        6.  **AVOID:** Generic greetings ("Hope you're well"), vague statements ("My background aligns well"), direct demands ("Please refer me"), lengthy introductions or explanations.
        7.  **Output ONLY the message text.** No "Hi [Name]," prefix, no subject line, no signature block, just the raw message content suitable for the LinkedIn connection note field.

        **Connection Reason Example (Use/Adapt):** {connection_reason}
        """

        logging.info("Generating LinkedIn Referral Message (Strict Length)...")
        # Use the helper function, passing the API key
        message_result = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=GENERATION_MODEL_NAME, # Use appropriate generation model
            temperature=0.55, # Slightly lower temp for consistency and conciseness
            stream=False # Expect a single string result
        )

        # --- Process Result ---
        if isinstance(message_result, str):
            message = message_result.strip() # Start with the stripped message

            # --- Strict Post-Processing Length Check & Truncation ---
            max_len = 300
            if len(message) > max_len:
                original_len = len(message)
                logging.warning(f"Generated LinkedIn message > {max_len} chars ({original_len}). Performing truncation.")

                # Attempt to truncate cleanly at sentence endings or spaces
                cut_off_point = -1
                # Prioritize sentence-ending punctuation, searching backwards from max_len
                for char in ['.', '?', '!']:
                    point = message[:max_len].rfind(char)
                    # Ensure the break point isn't too early (e.g., must be after char 200)
                    min_reasonable_len = max(max_len - 100, 0) # Ensure minimum reasonable length
                    if point > max(cut_off_point, min_reasonable_len):
                        cut_off_point = point

                # If no sentence end found, try common punctuation like comma
                if cut_off_point == -1:
                     for char in [',', ';']:
                           point = message[:max_len].rfind(char)
                           min_reasonable_len = max(max_len - 100, 0)
                           if point > max(cut_off_point, min_reasonable_len):
                                cut_off_point = point

                # If still no good punctuation break, try the last space
                if cut_off_point == -1:
                     point = message[:max_len].rfind(' ')
                     min_reasonable_len = max(max_len - 100, 0)
                     if point > max(cut_off_point, min_reasonable_len):
                          cut_off_point = point

                # Perform the truncation
                if cut_off_point != -1:
                    # Truncate at the identified point (+1 to include the char)
                    message = message[:cut_off_point+1].rstrip() # Trim trailing space if cut on space

                    # Add ellipsis if space permits and not already ending punctuation
                    if message[-1] not in ['.','?','!'] and len(message) + 3 <= max_len:
                         message += "..."
                    # If ellipsis doesn't fit, just use the truncated message up to max_len
                    elif len(message) > max_len:
                         message = message[:max_len]

                    logging.info(f"Cleanly truncated message to {len(message)} chars.")
                else:
                    # Hard truncate if no suitable break point found near the end
                    message = message[:max_len-3] + "..."
                    logging.info(f"Hard truncated message to {len(message)} chars.")

            # Final length check just in case truncation logic had issues
            if len(message) > max_len:
                 message = message[:max_len]
                 logging.warning(f"Message still exceeded max_len after truncation attempts. Forced to {len(message)} chars.")

            logging.info(f"Final LinkedIn message length: {len(message)} chars.")
            return message
        else:
             # Handle unexpected return type (e.g., if helper returned generator or None)
             logging.error(f"Unexpected return type from _call_llm_async for LinkedIn message: {type(message_result)}")
             # Provide a user-facing error message
             return f"Error: Could not generate message (unexpected result type)."


    except Exception as e:
        logging.error(f"Error generating LinkedIn message: {e}", exc_info=True)
        # Return a user-friendly error message, maybe truncate the exception details
        return f"Error generating message. Details: {str(e)[:150]}"


# --- Enhanced Resume Parsing (Sections, Keywords, Actions/Results) ---
async def parse_resume_advanced_llm(resume_content: str, google_api_key: str) -> Dict[str, Any]:
    """
    Parses resume into sections, extracts keywords, and identifies action verbs + quantifiable results.
    Returns dict containing 'sections', 'extracted_keywords', 'achievements', and 'error' (None on success).
    """
    prompt = f"""
    Analyze the following resume text. Perform three tasks:
    1.  **Parse Sections:** Divide the resume into logical sections based on standard headings (e.g., Summary, Skills, Experience, Education, Projects, Certifications, Awards). Use lowercase standardized keys for the output JSON (e.g., "summary", "skills", "experience"). Preserve the original text content within each identified section. If sections are unclear or non-standard, group them under an 'other' key or make a best guess based on content. Include a "full_text" key containing the entire original resume.
    2.  **Extract Keywords:** Identify and extract a comprehensive list of specific technical skills (languages, frameworks, tools, platforms, databases), methodologies (Agile, Scrum), certifications, and relevant soft skills (communication, leadership, teamwork) mentioned anywhere in the resume. Output should be a list of unique strings, all lowercase.
    3.  **Extract Achievements:** From the 'experience' section ONLY, identify bullet points or phrases describing accomplishments. For each, extract an object containing:
        * 'action_verb': The primary action verb starting the bullet/phrase (normalized to base form, e.g., "Managed" from "Managing").
        * 'quantifiable_result': The part of the phrase describing a measurable outcome (e.g., "budget of $5M", "efficiency by 15%", "over 100 users"). Set to null if no clear quantifiable metric is found in that specific bullet/phrase. Focus on bullets clearly starting with verbs.

    Return ONLY a valid JSON object matching this structure:
    {{
        "sections": {{
            "full_text": "...",
            "summary": "...",
            "skills": "...",
            "experience": "...",
            "education": "...",
            // other sections as found...
        }},
        "extracted_keywords": ["python", "java", "aws", "docker", "agile", "communication", ...],
        "achievements": [
            {{"action_verb": "Managed", "quantifiable_result": "budget of $5M"}},
            {{"action_verb": "Increased", "quantifiable_result": "efficiency by 15%"}},
            {{"action_verb": "Developed", "quantifiable_result": null}},
            // ... other achievements
        ]
    }}

    Resume Text to Analyze:
    ---
    {resume_content}
    ---

    JSON Output:
    """
    # Default structure in case of errors during the process
    error_result = {
        "sections": {"full_text": resume_content}, # Keep full text even on error
        "extracted_keywords": [],
        "achievements": [],
        "error": "Initialization error" # Placeholder, will be overwritten
    }
    try:
        # Use the helper function for the LLM call
        response_text = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=EXTRACTION_MODEL_NAME, # Use model suitable for extraction
            temperature=0.1, # Low temperature for factual extraction
            request_json=True # Request JSON output format
        )

        # --- Process LLM Response ---
        if isinstance(response_text, str):
            raw_json_string = response_text # Already cleaned by helper if possible

            # --- Parse and Validate JSON ---
            try:
                parsed_json = json.loads(raw_json_string)

                # --- Detailed Validation of Parsed Structure ---
                if not isinstance(parsed_json, dict):
                    raise ValueError("LLM did not return a dictionary.")

                # Validate 'sections' structure
                sections = parsed_json.get("sections")
                if not isinstance(sections, dict):
                     logging.warning("Invalid 'sections' structure received from LLM. Resetting to basic.")
                     sections = {"full_text": resume_content} # Fallback
                # Ensure 'full_text' key exists, add if missing
                if "full_text" not in sections:
                     sections["full_text"] = resume_content
                     logging.info("Added missing 'full_text' key to sections.")

                # Validate 'extracted_keywords' structure and content
                keywords_raw = parsed_json.get("extracted_keywords")
                if not isinstance(keywords_raw, list):
                    logging.warning("Invalid 'extracted_keywords' structure received. Resetting to empty list.")
                    keywords = []
                else:
                    # Clean, lowercase, deduplicate, and sort keywords
                    keywords = sorted(list(set([
                        str(kw).lower().strip()
                        for kw in keywords_raw if isinstance(kw, str) and str(kw).strip() # Ensure string and not empty
                    ])))

                # Validate 'achievements' structure and content
                achievements_raw = parsed_json.get("achievements")
                if not isinstance(achievements_raw, list):
                    logging.warning("Invalid 'achievements' structure received. Resetting to empty list.")
                    achievements = []
                else:
                    achievements = []
                    for item in achievements_raw:
                        # Check if item is a dict with required keys
                        if isinstance(item, dict) and 'action_verb' in item:
                            action_verb = str(item.get("action_verb", "")).strip()
                            # Only add if action_verb is not empty
                            if action_verb:
                                quant_result_raw = item.get("quantifiable_result")
                                # Store result as string if present and not empty, else None
                                quant_result = str(quant_result_raw).strip() if quant_result_raw is not None and str(quant_result_raw).strip() else None

                                achievements.append({
                                    "action_verb": action_verb,
                                    "quantifiable_result": quant_result
                                })
                        else:
                            logging.warning(f"Skipping invalid achievement item structure: {item}")

                # --- Success ---
                logging.info("Resume sections, keywords, and achievements parsed and validated successfully.")
                return {
                    "sections": sections,
                    "extracted_keywords": keywords,
                    "achievements": achievements,
                    "error": None # Explicitly set error to None on success
                }

            # --- Handle JSON Parsing/Validation Errors ---
            except json.JSONDecodeError as json_err:
                err_pos = json_err.pos
                err_msg = f"Failed to parse resume JSON response: {json_err} at position {err_pos}."
                problem_char_info = "Could not identify problematic character."
                snippet_info = f"JSON Snippet (around pos {err_pos}): '{raw_json_string[max(0, err_pos-40):min(len(raw_json_string), err_pos+40)]}'"

                # Attempt to pinpoint the problematic character
                if 0 <= err_pos < len(raw_json_string):
                    try:
                        problem_char = raw_json_string[err_pos]
                        problem_char_code = ord(problem_char)
                        problem_char_info = f"Problematic character: '{repr(problem_char)}' (Unicode Code Point: {problem_char_code}, Hex: {hex(problem_char_code)})"
                    except Exception as char_e:
                        problem_char_info = f"Error accessing character at position {err_pos}: {char_e}"
                elif err_pos == len(raw_json_string):
                     problem_char_info = "Error occurred at the very end of the string (potentially incomplete JSON)."

                logging.error(f"{err_msg} {problem_char_info}. {snippet_info}")
                # Optionally log more context for debugging:
                # logging.debug(f"Full raw JSON string that failed parsing:\n{raw_json_string}")

                error_result["error"] = f"Invalid JSON response for resume parsing: {json_err} (Pos: {err_pos}). {problem_char_info}"
                error_result["raw_response_snippet"] = raw_json_string[max(0, err_pos-50):min(len(raw_json_string), err_pos+50)]
                return error_result

            except ValueError as val_err: # Catch validation errors raised within the try block
                 logging.error(f"Validation error during resume JSON processing: {val_err}", exc_info=True)
                 error_result["error"] = f"Error validating parsed resume JSON structure: {val_err}"
                 # Include a snippet of the raw JSON that caused validation failure
                 error_result["raw_response_snippet"] = raw_json_string[:500]
                 return error_result

            except Exception as e_inner: # Catch any other unexpected errors during processing
                logging.error(f"Unexpected error during resume JSON processing: {e_inner}", exc_info=True)
                error_result["error"] = f"Unexpected error processing resume JSON structure: {e_inner}"
                error_result["raw_response_snippet"] = raw_json_string[:500]
                return error_result
        else:
             # Handle case where LLM call did not return a string
             logging.error(f"Unexpected return type from _call_llm_async for resume parsing: {type(response_text)}")
             error_result["error"] = "Error: LLM call for resume parsing did not return text."
             return error_result

    # --- Handle Outer LLM Call Errors ---
    except Exception as e:
        logging.error(f"Error during LLM call for advanced resume parsing: {e}", exc_info=True)
        error_result["error"] = f"Advanced Resume Parsing LLM Call Error: {e}"
        return error_result


# --- Enhanced JD Extraction (including Company Context) ---
async def _extract_structured_data_enhanced_jd(text: str, google_api_key: str) -> Dict[str, Any]:
    """Extracts structured data from JD, including company context, with improved validation."""
    schema = {
        "job_title": "string (e.g., 'Senior Software Engineer')",
        "company_name": "string (e.g., 'Tech Innovations Ltd.')",
        "key_skills_requirements": ["string (List distinct technical skills like 'Python', 'AWS', tools like 'Docker', platforms like 'Linux', required experience like '5+ years Java', AND explicitly mentioned soft skills like 'strong communication', 'teamwork', 'leadership')"],
        "location": "string (e.g., 'Remote', 'San Francisco, CA', 'Hybrid - London')",
        "summary": "string (A brief, 1-2 sentence high-level summary or objective of the role as described in the JD)",
        "company_values_mission_challenges": "string (Concise notes capturing mentioned company culture points like 'fast-paced environment', stated mission/values, or specific challenges/goals the role addresses, e.g., 'scaling our platform', 'driving innovation in AI')"
    }
    instructions = "Carefully analyze the Job Description text provided. Extract the requested information accurately, adhering to the specified format for each field. For 'key_skills_requirements', create a comprehensive list including technical skills (be specific, e.g., 'Python 3.x', 'React.js'), tools/platforms ('Kubernetes', 'Azure DevOps'), specific experience levels ('minimum 3 years'), and any explicitly stated soft skills ('excellent presentation skills', 'ability to lead projects'). For 'company_values_mission_challenges', capture the essence of the company's self-description regarding its work environment, goals, values, or the problems this role is intended to solve. Return null if information for a field cannot be found."

    prompt = f"""
    Analyze the following Job Description and extract information according to the schema provided below.
    Return ONLY a valid JSON object adhering strictly to the schema structure. Use null for any field where information is not present in the text.

    Instructions: {instructions}

    Schema:
    ```json
    {json.dumps(schema, indent=2)}
    ```

    Job Description Text:
    ---
    {text}
    ---

    JSON Output:
    """
    # Default result structure
    result = {"data": {}, "error": None}

    try:
        # Use the helper function for the LLM call
        response_text = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=EXTRACTION_MODEL_NAME,
            temperature=0.1, # Low temperature for accurate extraction
            request_json=True # Request JSON output
        )

        # --- Process LLM Response ---
        if isinstance(response_text, str):
            raw_json_string = response_text # Assumed cleaned by helper

            # --- Parse and Validate JSON ---
            try:
                extracted_data = json.loads(raw_json_string)

                if not isinstance(extracted_data, dict):
                    raise ValueError("LLM response for JD extraction was not a dictionary.")

                # --- Schema Validation and Normalization ---
                validated_data = {}
                expected_keys = schema.keys()

                for key in expected_keys:
                    if key not in extracted_data:
                         logging.warning(f"JD Extraction: Key '{key}' missing from LLM response. Setting to null.")
                         validated_data[key] = None
                         continue # Skip further checks for this missing key

                    value = extracted_data[key]

                    # Validate/normalize 'key_skills_requirements' specifically
                    if key == "key_skills_requirements":
                        if value is None:
                             validated_data[key] = []
                        elif isinstance(value, list):
                             # Ensure all items are non-empty strings
                             validated_data[key] = [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()]
                        elif isinstance(value, str):
                             # Attempt to split if it's a single string
                             logging.warning(f"JD Extraction: '{key}' was a string, attempting to split.")
                             validated_data[key] = [item.strip() for item in re.split(r'[,\n;•*-]', value) if item.strip()]
                        else:
                             logging.warning(f"JD Extraction: Invalid type for '{key}' ({type(value)}). Setting to empty list.")
                             validated_data[key] = []
                    # Validate other fields are strings or null
                    elif key in ["job_title", "company_name", "location", "summary", "company_values_mission_challenges"]:
                        if value is None or isinstance(value, str):
                             validated_data[key] = value.strip() if isinstance(value, str) else None
                        else:
                             logging.warning(f"JD Extraction: Invalid type for '{key}' ({type(value)}). Setting to null.")
                             validated_data[key] = None
                    else:
                         # Should not happen if schema keys match code, but handle defensively
                         validated_data[key] = value

                logging.info("Enhanced structured data extracted and validated for JD.")
                result["data"] = validated_data
                result["error"] = None

            # --- Handle JSON Parsing/Validation Errors ---
            except (json.JSONDecodeError, ValueError) as json_err:
                 error_msg = f"Failed to parse/validate JD JSON response: {json_err}"
                 logging.error(f"{error_msg}. Raw Response Snippet: {raw_json_string[:500]}...")
                 result["error"] = error_msg
                 result["data"] = {} # Ensure data is empty on error
            except Exception as e_inner:
                 error_msg = f"Unexpected error processing JD JSON: {e_inner}"
                 logging.error(error_msg, exc_info=True)
                 result["error"] = error_msg
                 result["data"] = {}
        else:
             # Handle case where LLM call did not return a string
             logging.error(f"Unexpected return type from _call_llm_async for JD extraction: {type(response_text)}")
             result["error"] = "Error: LLM call for JD extraction did not return text."
             result["data"] = {}

    # --- Handle Outer LLM Call Errors ---
    except Exception as e:
        logging.error(f"Error during LLM call for enhanced JD extraction: {e}", exc_info=True)
        result["error"] = f"Enhanced JD Extraction LLM Call Error: {e}"
        result["data"] = {}

    return result


# --- Rank JD Requirements (Async LLM) ---
async def _rank_jd_requirements(requirements: List[str], google_api_key: str) -> Dict[str, Any]:
    """Uses an LLM to rank JD requirements provided as a list of strings (async)."""

    # Input Validation
    if not isinstance(requirements, list):
         logging.warning(f"Input 'requirements' for ranking is not a list ({type(requirements)}). Returning empty.")
         # Return original input if not list? No, return empty as ranking isn't possible.
         return {"ranked_list": [], "error": "Input requirements was not a list."}

    # Filter out non-string or empty string items
    valid_requirements = [str(req).strip() for req in requirements if isinstance(req, str) and str(req).strip()]

    if not valid_requirements:
         logging.info("No valid, non-empty requirements provided to rank.")
         return {"ranked_list": [], "error": None} # Not an error, just nothing to do.

    # Format requirements for the prompt
    req_string = "\n".join(f"- {req}" for req in valid_requirements)

    prompt = f"""
    Analyze the following list of job requirements extracted from a job description. Your task is to rank these requirements based on their likely importance from the perspective of a hiring manager or ATS screening.
    Prioritize mandatory qualifications (e.g., specific degree, years of experience in a core technology, essential certifications) and core technical skills highest. Then rank key responsibilities and important soft skills. Preferred qualifications or general duties should be ranked lower.

    Return ONLY a valid JSON object containing a single key: "ranked_list".
    The value associated with "ranked_list" MUST be a list of strings, where each string is one of the original requirement texts provided below. The list MUST contain the exact same requirements as the input, but ordered from MOST important to LEAST important based on your analysis.

    Requirements List to Rank:
    ---
    {req_string}
    ---

    JSON Output (containing ONLY the "ranked_list" key with the reordered list of strings):
    """
    # Default result structure
    result = {"ranked_list": valid_requirements, "error": None} # Default to original list

    try:
         # Use the helper function for the LLM call
        response_text = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=EXTRACTION_MODEL_NAME, # Use a model good for analysis/extraction
            temperature=0.15, # Low temperature for more deterministic ranking
            request_json=True # Request JSON output
        )

        # --- Process LLM Response ---
        if isinstance(response_text, str):
            raw_json_string = response_text # Assumed cleaned by helper

            # --- Parse and Validate JSON ---
            try:
                ranked_data = json.loads(raw_json_string)

                # Validate structure
                if not isinstance(ranked_data, dict) or "ranked_list" not in ranked_data:
                    raise ValueError("LLM response missing 'ranked_list' key or is not a dictionary.")

                ranked_list_raw = ranked_data["ranked_list"]
                if not isinstance(ranked_list_raw, list):
                    raise ValueError("Value associated with 'ranked_list' is not a list.")

                # Validate content: ensure items are strings and try to match input count
                validated_ranked_list = [str(item).strip() for item in ranked_list_raw if isinstance(item, str) and str(item).strip()]

                # Sanity Check: Compare length and content (optional but recommended)
                if len(validated_ranked_list) != len(valid_requirements):
                     logging.warning(f"Ranked list length ({len(validated_ranked_list)}) differs from input requirements length ({len(valid_requirements)}). LLM might have omitted or added items.")
                     # Decide how to handle: Use the potentially flawed list, or fallback? Let's use it but log warning.
                     result["ranked_list"] = validated_ranked_list # Use the list from LLM despite length mismatch
                elif set(validated_ranked_list) != set(valid_requirements):
                     logging.warning("Ranked list contains different items than the input requirements. Using LLM output.")
                     result["ranked_list"] = validated_ranked_list # Use the list from LLM despite content mismatch
                else:
                     # Length and content seem okay
                     result["ranked_list"] = validated_ranked_list
                     logging.info("JD Requirements ranked successfully.")

                result["error"] = None # Success

            # --- Handle JSON Parsing/Validation Errors ---
            except (json.JSONDecodeError, ValueError) as json_err:
                 error_msg = f"Failed to parse/validate JD ranking JSON response: {json_err}"
                 logging.error(f"{error_msg}. Raw Response Snippet: {raw_json_string[:500]}...")
                 result["error"] = error_msg
                 # Keep default ranked_list (original order) on error
            except Exception as e_inner:
                 error_msg = f"Unexpected error processing JD ranking JSON: {e_inner}"
                 logging.error(error_msg, exc_info=True)
                 result["error"] = error_msg
                 # Keep default ranked_list on error
        else:
            # Handle case where LLM call did not return a string
            logging.error(f"Unexpected return type from _call_llm_async for JD ranking: {type(response_text)}")
            result["error"] = "Error: LLM call for JD ranking did not return text."
            # Keep default ranked_list on error

    # --- Handle Outer LLM Call Errors ---
    except Exception as e:
        logging.error(f"Error during LLM call for JD requirement ranking: {e}", exc_info=True)
        result["error"] = f"JD Ranking LLM Call Error: {e}"
        # Keep default ranked_list on error

    return result


async def _select_top_bullets_llm(
    bullets: List[str],
    job_requirements: List[str],
    google_api_key: str,
    model_name: str = EXTRACTION_MODEL_NAME,
    max_bullets: int = 8
) -> Dict[str, Any]:
    """
    Uses an LLM to select the top N most relevant bullets from a list (usually from one job entry)
    based on job requirements.
    """
    if not bullets:
        return {"selected_list": [], "error": "No bullets provided for selection."}
    if not job_requirements:
        logging.warning("No job requirements provided for bullet selection. Returning original bullets (up to max).")
        # Return original bullets up to the limit, no error reported upstream
        return {"selected_list": bullets[:max_bullets], "error": None}

    # Format inputs for the prompt
    bullets_str = "\n".join(f"- {b}" for b in bullets)
    # Use only top N requirements for context to keep prompt manageable? Let's use all for now.
    requirements_str = "\n".join(f"- {r}" for r in job_requirements)

    prompt = f"""
    **Task:** Analyze the following 'List of Resume Bullet Points' and select the ones most relevant to the 'Target Job Requirements'.

    **Context:**
    * **Target Job Requirements (Ranked by Importance):**
    {requirements_str}
    * **List of Resume Bullet Points (from a single job):**
    {bullets_str}

    **Instructions:**
    1.  Carefully compare each bullet point from the 'List of Resume Bullet Points' against the 'Target Job Requirements'.
    2.  Identify and select **up to {max_bullets}** bullet points from the *original list* that demonstrate the strongest alignment and relevance to the *most important* job requirements.
    3.  Prioritize bullets that showcase skills, experiences, or quantifiable results directly mentioned or implied in the job requirements.
    4.  Return ONLY a valid JSON object containing a single key: "selected_bullets".
    5.  The value of "selected_bullets" MUST be a list containing the exact text of the selected bullet points, ordered from most relevant to least relevant according to your analysis.
    6.  The list MUST contain **no more than {max_bullets}** bullet points. If the original list has fewer than {max_bullets} relevant bullets, return only the relevant ones. If the original list itself has fewer than {max_bullets} bullets, return all of them if they seem relevant.

    **JSON Output (containing ONLY the "selected_bullets" key):**
    """

    # Default to original N bullets in case of failure below
    result = {"selected_list": bullets[:max_bullets], "error": None}

    try:
        response_text = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=model_name,
            temperature=0.1, # Low temp for deterministic selection
            request_json=True
        )

        if isinstance(response_text, str):
            raw_json_string = response_text
            try:
                selection_data = json.loads(raw_json_string)
                if not isinstance(selection_data, dict) or "selected_bullets" not in selection_data:
                    raise ValueError("LLM response missing 'selected_bullets' key or is not a dictionary.")

                selected_list_raw = selection_data["selected_bullets"]
                if not isinstance(selected_list_raw, list):
                    raise ValueError("Value associated with 'selected_bullets' is not a list.")

                # Validate content: ensure items are strings and cap the length
                validated_selected_list = [
                    str(item).strip() for item in selected_list_raw
                    if isinstance(item, str) and str(item).strip()
                ][:max_bullets] # Ensure max length constraint is applied *after* LLM potentially violates it

                # More robust check: Ensure selected bullets were actually present in the input list
                original_bullets_set = set(bullets)
                final_selection = []
                for b_selected in validated_selected_list:
                    # Simple check for exact match
                    if b_selected in original_bullets_set:
                        final_selection.append(b_selected)
                    else:
                        logging.warning(f"LLM selected bullet slightly differs or wasn't in original list: '{b_selected}'")
                        # Discarding slightly modified bullets for strictness.

                if len(final_selection) != len(validated_selected_list):
                    logging.warning(f"Filtered LLM selection to bullets found in the original list. Original selection count: {len(validated_selected_list)}, Final count: {len(final_selection)}")

                if not final_selection: # Handle case where LLM hallucinates or returns nothing valid
                    logging.warning("LLM returned no valid bullets matching the original list. Falling back to top N original bullets.")
                    result["selected_list"] = bullets[:max_bullets] # Fallback to original list (first N)
                    result["error"] = "LLM failed to select valid bullets matching originals." # Report specific issue
                else:
                    result["selected_list"] = final_selection
                    result["error"] = None # Success
                    logging.info(f"Successfully selected {len(final_selection)} relevant bullets.")


            except (json.JSONDecodeError, ValueError) as json_err:
                error_msg = f"Failed to parse/validate bullet selection JSON: {json_err}"
                logging.error(f"{error_msg}. Raw Response: {raw_json_string[:500]}...")
                result["error"] = error_msg
                # Keep default selected_list (top N original) on error
            except Exception as e_inner:
                error_msg = f"Unexpected error processing bullet selection JSON: {e_inner}"
                logging.error(error_msg, exc_info=True)
                result["error"] = error_msg
                # Keep default selected_list on error
        else:
             # Handle non-string response from LLM call
             logging.error(f"Unexpected return type from _call_llm_async for bullet selection: {type(response_text)}")
             result["error"] = "Error: LLM call for bullet selection did not return text."
             # Keep default selected_list on error

    except Exception as e:
        # Handle errors during the LLM call itself
        logging.error(f"Error during LLM call for bullet selection: {e}", exc_info=True)
        result["error"] = f"Bullet Selection LLM Call Error: {e}"
        # Keep default selected_list on error

    return result


# --- Select Top Projects based on JD (Async LLM) ---
async def _select_top_projects_llm(
    projects: List[str],               # Assumes a list of project description strings
    job_requirements: List[str],
    google_api_key: str,
    model_name: str = EXTRACTION_MODEL_NAME,
    max_projects: int = 3              # Default to selecting top 3
) -> Dict[str, Any]:
    """
    Uses an LLM to select the top 3 most relevant projects from a list based on job requirements.
    Assumes each project in the list is a string description.
    """
    if not projects:
        return {"selected_list": [], "error": "No projects provided for selection."}
    if not job_requirements:
        logging.warning("No job requirements provided for project selection. Returning original projects (up to max).")
        # Return original projects up to the limit, no error reported upstream
        return {"selected_list": projects[:max_projects], "error": None}

    # Format inputs for the prompt
    projects_str = "\n".join(f"- {p}" for p in projects)
    requirements_str = "\n".join(f"- {r}" for r in job_requirements)

    prompt = f"""
    **Task:** Analyze the following 'List of Resume Projects' and select the ones most relevant to the 'Target Job Requirements'.

    **Context:**
    * **Target Job Requirements (Ranked by Importance):**
    {requirements_str}
    * **List of Resume Projects (Descriptions):**
    {projects_str}

    **Instructions:**
    1.  Carefully compare each project description from the 'List of Resume Projects' against the 'Target Job Requirements'.
    2.  Identify and select **up to {max_projects}** projects from the *original list* that demonstrate the strongest alignment and relevance to the *most important* job requirements.
    3.  Prioritize projects that showcase skills, technologies, or outcomes directly mentioned or implied in the job requirements.
    4.  Return ONLY a valid JSON object containing a single key: "selected_projects".
    5.  The value of "selected_projects" MUST be a list containing the exact text of the selected project descriptions, ordered from most relevant to least relevant according to your analysis.
    6.  The list MUST contain **no more than {max_projects}** projects. If the original list has fewer than {max_projects} relevant projects, return only the relevant ones. If the original list itself has fewer than {max_projects} projects, return all of them if they seem relevant.

    **JSON Output (containing ONLY the "selected_projects" key):**
    """

    # Default to original N projects in case of failure below
    result = {"selected_list": projects[:max_projects], "error": None}

    try:
        response_text = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=model_name,
            temperature=0.1, # Low temp for deterministic selection
            request_json=True
        )

        if isinstance(response_text, str):
            raw_json_string = response_text
            try:
                selection_data = json.loads(raw_json_string)
                if not isinstance(selection_data, dict) or "selected_projects" not in selection_data:
                    raise ValueError("LLM response missing 'selected_projects' key or is not a dictionary.")

                selected_list_raw = selection_data["selected_projects"]
                if not isinstance(selected_list_raw, list):
                    raise ValueError("Value associated with 'selected_projects' is not a list.")

                # Validate content: ensure items are strings and cap the length
                validated_selected_list = [
                    str(item).strip() for item in selected_list_raw
                    if isinstance(item, str) and str(item).strip()
                ][:max_projects] # Ensure max length constraint is applied *after* LLM potentially violates it

                # More robust check: Ensure selected projects were actually present in the input list
                original_projects_set = set(projects)
                final_selection = []
                for p_selected in validated_selected_list:
                    # Simple check for exact match
                    if p_selected in original_projects_set:
                        final_selection.append(p_selected)
                    else:
                        logging.warning(f"LLM selected project slightly differs or wasn't in original list: '{p_selected}'")
                        # Discarding slightly modified projects for strictness.

                if len(final_selection) != len(validated_selected_list):
                    logging.warning(f"Filtered LLM selection to projects found in the original list. Original selection count: {len(validated_selected_list)}, Final count: {len(final_selection)}")

                if not final_selection: # Handle case where LLM hallucinates or returns nothing valid
                    logging.warning("LLM returned no valid projects matching the original list. Falling back to top N original projects.")
                    result["selected_list"] = projects[:max_projects] # Fallback to original list (first N)
                    result["error"] = "LLM failed to select valid projects matching originals." # Report specific issue
                else:
                    result["selected_list"] = final_selection
                    result["error"] = None # Success
                    logging.info(f"Successfully selected {len(final_selection)} relevant projects.")


            except (json.JSONDecodeError, ValueError) as json_err:
                error_msg = f"Failed to parse/validate project selection JSON: {json_err}"
                logging.error(f"{error_msg}. Raw Response: {raw_json_string[:500]}...")
                result["error"] = error_msg
                # Keep default selected_list (top N original) on error
            except Exception as e_inner:
                error_msg = f"Unexpected error processing project selection JSON: {e_inner}"
                logging.error(error_msg, exc_info=True)
                result["error"] = error_msg
                # Keep default selected_list on error
        else:
             # Handle non-string response from LLM call
             logging.error(f"Unexpected return type from _call_llm_async for project selection: {type(response_text)}")
             result["error"] = "Error: LLM call for project selection did not return text."
             # Keep default selected_list on error

    except Exception as e:
        # Handle errors during the LLM call itself
        logging.error(f"Error during LLM call for project selection: {e}", exc_info=True)
        result["error"] = f"Project Selection LLM Call Error: {e}"
        # Keep default selected_list on error

    return result


# --- Data Preparation (Async) ---
async def _prepare_common_data(job_description: str, resume_content: str, google_api_key: str) -> Dict[str, Any]:
    """Orchestrates enhanced data extraction (JD), parsing (Resume), ranking (JD reqs), and keyword comparison."""
    # Initialize results with defaults and include raw inputs
    results = {
        "error": None,
        "jd_data": {}, # Will hold structured JD { "data": {...}, "error": ... }
        "resume_data": {}, # Will hold structured Resume { "sections": ..., "keywords": ..., "achievements": ..., "error": ... }
        "ranked_jd_requirements": [], # List of ranked requirement strings
        "missing_keywords_from_resume": [], # List of keywords in JD but not Resume
        "raw_resume": resume_content,
        "raw_jd": job_description,
        # Store intermediate results for debugging if needed
        "_debug_jd_extraction_result": None,
        "_debug_resume_parsing_result": None,
        "_debug_ranking_result": None,
    }
    task_errors = [] # Collect errors from sequential steps

    try:
        # --- STEP 1: Extract structured data from Job Description ---
        logging.info("Data Prep Step 1: Extracting structured data from Job Description...")
        jd_extraction_result = await _extract_structured_data_enhanced_jd(job_description, google_api_key)
        results["_debug_jd_extraction_result"] = jd_extraction_result # Store for debug
        if jd_extraction_result.get("error"):
            task_errors.append(f"JD Extraction failed: {jd_extraction_result['error']}")
            results["jd_data"] = {} # Ensure default structure on error
        else:
            results["jd_data"] = jd_extraction_result.get("data", {}) # Store the actual data dict
            logging.info("Data Prep Step 1: JD Extraction successful.")


        # --- STEP 2: Parse Resume ---
        logging.info("Data Prep Step 2: Parsing Resume...")
        resume_parsing_result = await parse_resume_advanced_llm(resume_content, google_api_key)
        results["_debug_resume_parsing_result"] = resume_parsing_result # Store for debug
        if resume_parsing_result.get("error"):
            task_errors.append(f"Resume Parsing failed: {resume_parsing_result['error']}")
            # Keep raw text in sections if parsing fails, clear structured data
            results["resume_data"] = {
                "sections": resume_parsing_result.get("sections", {"full_text": resume_content}), # Try to keep sections
                "extracted_keywords": [],
                "achievements": [],
                "error": resume_parsing_result.get("error") # Propagate error info
            }
        else:
            # Store the successful parsing result directly
            results["resume_data"] = resume_parsing_result # Contains sections, keywords, achievements, error=None
            logging.info("Data Prep Step 2: Resume Parsing successful.")


        # --- STEP 3: Rank JD Requirements ---
        logging.info("Data Prep Step 3: Ranking JD Requirements...")
        # Rank requirements only if successfully extracted in Step 1
        extracted_requirements = results.get("jd_data", {}).get("key_skills_requirements", [])
        if extracted_requirements and isinstance(extracted_requirements, list):
             ranking_result = await _rank_jd_requirements(extracted_requirements, google_api_key)
             results["_debug_ranking_result"] = ranking_result # Store for debug
             if ranking_result.get("error"):
                  task_errors.append(f"JD Ranking error: {ranking_result['error']}")
                  # Keep the unranked list from JD extraction if ranking fails
                  results["ranked_jd_requirements"] = extracted_requirements
                  logging.warning(f"JD Ranking failed, using unranked requirements. Error: {ranking_result['error']}")
             else:
                  results["ranked_jd_requirements"] = ranking_result.get("ranked_list", [])
                  logging.info("Data Prep Step 3: JD Ranking successful.")
        else:
             # Handle cases where requirements weren't extracted or weren't a list
             results["ranked_jd_requirements"] = []
             if not extracted_requirements and "JD Extraction failed" not in " ".join(task_errors):
                  logging.warning("Skipping JD ranking because no requirements were successfully extracted.")
                  task_errors.append("JD Ranking skipped: No requirements extracted.")
             elif not isinstance(extracted_requirements, list):
                   logging.warning(f"Skipping JD ranking because extracted requirements were not a list ({type(extracted_requirements)}).")
                   task_errors.append("JD Ranking skipped: Requirements not a list.")
             # No need to add error if JD extraction already failed


        # --- STEP 4: Identify Missing Keywords ---
        logging.info("Data Prep Step 4: Identifying Missing Keywords (Resume vs JD)...")
        # Compare only if JD requirements and resume keywords were successfully extracted
        jd_keywords_list_for_comp = results.get("jd_data", {}).get("key_skills_requirements", []) # Use original extracted list
        resume_keywords_list_for_comp = results.get("resume_data", {}).get("extracted_keywords", []) # Get from resume_data dict

        # Check if both lists are valid and non-empty
        valid_jd_keywords = isinstance(jd_keywords_list_for_comp, list) and jd_keywords_list_for_comp
        valid_resume_keywords = isinstance(resume_keywords_list_for_comp, list) # Allow empty resume list

        if valid_jd_keywords and valid_resume_keywords is not None: # Check resume list validity (can be empty)
             try:
                  # Ensure JD keywords are strings and lowercased for comparison set
                  # Handle potential non-string items defensively
                  jd_keywords_set = set(
                       str(kw).lower().strip() for kw in jd_keywords_list_for_comp
                       if isinstance(kw, (str, int, float)) and str(kw).strip()
                  )
                  # Resume keywords are already lowercased list from parsing function
                  resume_keywords_set = set(resume_keywords_list_for_comp) # Assumes list of strings

                  missing_keywords = sorted(list(jd_keywords_set - resume_keywords_set))
                  results["missing_keywords_from_resume"] = missing_keywords
                  logging.info(f"Data Prep Step 4: Identified {len(missing_keywords)} potentially missing keywords.")
             except Exception as kw_err:
                  task_errors.append(f"Keyword comparison error: {kw_err}")
                  results["missing_keywords_from_resume"] = [] # Reset on error
                  logging.error(f"Error during keyword comparison: {kw_err}", exc_info=True)
        else:
             results["missing_keywords_from_resume"] = []
             if not valid_jd_keywords:
                  logging.info("Skipping missing keyword identification: JD keywords missing or invalid.")
                  task_errors.append("Keyword Comparison skipped: JD keywords unavailable.")
             elif valid_resume_keywords is None: # Check if resume keywords failed extraction severely
                  logging.info("Skipping missing keyword identification: Resume keywords unavailable.")
                  task_errors.append("Keyword Comparison skipped: Resume keywords unavailable.")


        # --- Finalize Errors ---
        if task_errors:
            results["error"] = "; ".join(task_errors)
            logging.error(f"Data preparation completed with errors: {results['error']}")
        else:
             logging.info("Data preparation completed successfully.")

    except Exception as e:
        # Catch-all for unexpected errors during the orchestration logic itself
        critical_error_msg = f"Critical Error in Data Preparation Pipeline: {e}"
        results["error"] = critical_error_msg
        logging.error(critical_error_msg, exc_info=True)
        # Ensure essential raw data is still present
        results["raw_resume"] = resume_content
        results["raw_jd"] = job_description
        # Clear potentially inconsistent intermediate data
        results["jd_data"] = {}
        results["resume_data"] = {}
        results["ranked_jd_requirements"] = []
        results["missing_keywords_from_resume"] = []


    return results


# --- Generator with Multi-Turn Refinement & Enhanced Instructions (Async Stream) ---
async def generate_application_text_streamed(
    name: str, email: str, job_description: str, resume_content: str,
    generation_type: str, google_api_key: str, tone: str
) -> AsyncGenerator[str, None]:
    """
    Generates Resume or Cover Letter using a multi-turn refinement process (async stream).
    - Cover Letter focuses on deep company alignment.
    - Resume focuses on strict ATS optimization rules, high keyword coverage (~80%),
      natural distribution, AND selecting top relevant experience/project points.
    """
    common_data = {} # To store results from _prepare_common_data
    selected_bullets_context_str = "" # Initialize context strings
    selected_projects_context_str = ""

    try:
        # --- 1. Prepare Data (JD Extraction, Resume Parsing, Ranking, Keyword Gap) ---
        yield "--- Preparing and analyzing inputs (Job Description, Resume)... ---\n"
        common_data = await _prepare_common_data(job_description, resume_content, google_api_key)

        # --- Handle Data Preparation Errors ---
        if common_data.get("error"):
            yield f"\n--- ERROR during data preparation: {common_data['error']} ---\n"
            # Provide more specific debug info based on error messages
            if "JD Extraction failed" in common_data["error"]:
                yield f"DEBUG: Could not fully process the Job Description. Check its format and content.\n"
            if "Resume Parsing failed" in common_data["error"]:
                yield f"DEBUG: Could not fully process the Resume. Check its format and content.\n"
                yield f"DEBUG: Raw Resume Snippet (first 500 chars):\n------\n{resume_content[:500]}\n------\n"
            yield "\n--- Generation stopped due to data preparation errors. ---\n"
            return # Stop generation if critical prep failed

        # --- Verify Essential Data Pieces ---
        jd_data = common_data.get("jd_data", {})
        if not jd_data or not jd_data.get("job_title") or not jd_data.get("key_skills_requirements"):
            yield "\n--- ERROR: Critical Job Description data (Title or Requirements) missing after preparation. Cannot proceed effectively. ---\n"
            yield f"DEBUG: Job Title: {jd_data.get('job_title')}\n"
            yield f"DEBUG: Requirements Count: {len(jd_data.get('key_skills_requirements', []))}\n"
            yield f"DEBUG: Job Title: {jd_data.get('job_title')}\n"
            yield f"DEBUG: Requirements Count: {len(jd_data.get('key_skills_requirements', []))}\n"            
            yield "\n--- Generation stopped due to missing critical JD data. ---\n"
            return

        resume_data = common_data.get("resume_data", {})
        if not resume_data or not resume_data.get("sections") or resume_data.get("extracted_keywords") is None:
            yield "\n--- WARNING: Resume data (Sections or Keywords) appears incomplete after preparation. Generation quality may be affected. ---\n"
            yield f"DEBUG: Resume Data Keys: {list(resume_data.keys())}\n"
            yield f"DEBUG: Resume Sections Keys: {list(resume_data.get('sections', {}).keys())}\n"
            yield f"DEBUG: Resume Keywords Count: {len(resume_data.get('extracted_keywords', [])) if resume_data.get('extracted_keywords') is not None else 'N/A'}\n"# ... (keep existing debug yields) ...
            # Decide whether to stop or continue. Let's continue but warn.

        # --- Extract common data points (needed for both selection and prompts) ---
        ranked_reqs = common_data.get("ranked_jd_requirements", [])
        resume_sections = resume_data.get("sections", {})
        resume_achievements = resume_data.get("achievements", []) # Used for bullet selection


        # --- *** NEW: Select Top Bullets & Projects (Only for Resumes) *** ---
        if generation_type == TYPE_RESUME:
            yield "--- Selecting top relevant experience and projects... ---\n"

            # --- Select Top Bullets (from Achievements) ---
            if resume_achievements and ranked_reqs:
                # Format achievements into strings for selection
                achievement_strings = []
                for ach in resume_achievements:
                    verb = ach.get('action_verb', 'Processed')
                    result = ach.get('quantifiable_result')
                    # Create a representative string (adjust format as needed)
                    ach_str = f"{verb}..." + (f" (Result: {result})" if result else "")
                    achievement_strings.append(ach_str)

                if achievement_strings:
                    logging.info(f"Attempting to select top bullets from {len(achievement_strings)} achievements.")
                    bullet_selection_result = await _select_top_bullets_llm(
                        bullets=achievement_strings,
                        job_requirements=ranked_reqs,
                        google_api_key=google_api_key,
                        max_bullets= 8 # Select slightly more bullets overall
                    )
                    if bullet_selection_result.get("error"):
                        logging.warning(f"Bullet selection failed: {bullet_selection_result['error']}. Proceeding without selected highlights.")
                    else:
                        selected_bullets = bullet_selection_result.get("selected_list", [])
                        if selected_bullets:
                            selected_bullets_context_str = "**Selected Experience Highlights (Prioritize incorporating these):**\n" + "\n".join(f"- {b}" for b in selected_bullets)
                            logging.info(f"Selected {len(selected_bullets)} experience highlights.")
                        else:
                            logging.info("Bullet selection ran but returned no bullets.")
                else:
                     logging.info("No formatted achievement strings to select bullets from.")

            else:
                logging.warning("Skipping bullet selection due to missing achievements or requirements.")


            # --- Select Top Projects ---
            projects_text = resume_sections.get("projects")
            if projects_text and isinstance(projects_text, str) and ranked_reqs:
                 # Attempt simple parsing: Split by potential project markers (e.g., double newline, heading)
                 # This is basic and might need refinement based on actual resume format.
                 # Assumes projects start with '**' or similar markdown heading-like structure potentially preceded by newlines.
                potential_projects = re.split(r'\n\s*\n(?=\*\*.*?\*\*\s*\n)', projects_text.strip()) # Split on blank lines before bold titles
                if len(potential_projects) <= 1 and '\n**' in projects_text: # Try splitting just on bold titles if the first split failed
                     potential_projects = re.split(r'(?<=\n)\*\*(.*?)\*\*\s*\n', projects_text.strip())
                     # Filter out empty strings and re-add markdown to titles
                     potential_projects = [f"**{p.strip()}**" for p in potential_projects if p and p.strip()]


                all_project_descriptions = [p.strip() for p in potential_projects if p and p.strip()]

                if all_project_descriptions:
                    logging.info(f"Attempting to select top projects from {len(all_project_descriptions)} potential project descriptions.")
                    project_selection_result = await _select_top_projects_llm(
                        projects=all_project_descriptions,
                        job_requirements=ranked_reqs,
                        google_api_key=google_api_key,
                        max_projects=3 # Select top 3
                    )
                    if project_selection_result.get("error"):
                        logging.warning(f"Project selection failed: {project_selection_result['error']}. Proceeding without selected projects.")
                    else:
                        selected_projects = project_selection_result.get("selected_list", [])
                        if selected_projects:
                            # Assume selected_projects are full markdown chunks
                            selected_projects_context_str = "**Selected Project Highlights (Prioritize incorporating these):**\n\n" + "\n\n".join(selected_projects)
                            logging.info(f"Selected {len(selected_projects)} project highlights.")
                        else:
                            logging.info("Project selection ran but returned no projects.")
                else:
                     logging.info("Could not parse project descriptions for selection.")

            else:
                 logging.warning("Skipping project selection due to missing project text or requirements.")
            yield "--- Selection finished. Proceeding with generation... ---\n"
        # --- *** END NEW SELECTION LOGIC *** ---


        # --- Format Common Inputs for Prompts (Now includes selected context) ---
        # Safely access data using .get() - Already done above for resume_sections, resume_keywords, resume_achievements, ranked_reqs
        resume_keywords = resume_data.get("extracted_keywords", [])
        resume_achievements = resume_data.get("achievements", [])

        # Format resume sections string for prompt context (handle truncation)
        # Use the original sections here, selected context is added separately below
        resume_sections_str = "\n\n".join(
            f"**{sec.upper()}**\n{content}"
            for sec, content in resume_sections.items()
            if content and sec != 'full_text' # Exclude the full_text key itself
        )
        MAX_RESUME_PROMPT_LEN = 8000 # Adjust based on model context window limits
        if len(resume_sections_str) > MAX_RESUME_PROMPT_LEN:
            resume_sections_str = resume_sections_str[:MAX_RESUME_PROMPT_LEN] + "\n... [Resume Sections Truncated] ..."
            logging.warning(f"Truncated resume sections string for prompt to {MAX_RESUME_PROMPT_LEN} chars.")
        elif not resume_sections_str:
             # Fallback to raw resume if parsed sections are empty
             raw_resume = common_data.get("raw_resume", "Resume content not available.")
             resume_sections_str = raw_resume[:MAX_RESUME_PROMPT_LEN] # Truncate raw too
             if len(raw_resume) > MAX_RESUME_PROMPT_LEN:
                 resume_sections_str += "\n... [Raw Resume Truncated] ..."
             logging.warning("Resume sections empty after parsing attempt, using raw resume content in prompt.")


        # Format other common data points
        ranked_reqs = common_data.get("ranked_jd_requirements", [])
        ranked_req_str = "\n".join(f"- {req}" for req in ranked_reqs) if ranked_reqs else "N/A - Ranking may have failed or no requirements found."
        jd_title = jd_data.get('job_title', 'N/A')
        jd_company = jd_data.get('company_name', 'N/A')
        company_context_jd = jd_data.get('company_values_mission_challenges', 'N/A')
        missing_keywords = common_data.get("missing_keywords_from_resume", [])
        missing_keywords_str = ", ".join(f"`{kw}`" for kw in missing_keywords) if missing_keywords else "None identified or comparison failed."

        # Safely format achievements sample as JSON string (still useful for general context)
        achievements_sample_str = "N/A"
        try:
            # Sample first 5 achievements for brevity in prompt
            achievements_sample_str = json.dumps(resume_achievements[:5], indent=2)
        except TypeError as json_e:
            logging.error(f"Could not serialize resume achievements for prompt: {json_e}")
            achievements_sample_str = "[Error serializing achievements]"


        # --- External Company Info Placeholder ---
        external_company_prompt_section = "**External Company Research Findings:** Not Available (Feature not implemented)."


        # --- Define Type-Specific Prompt Components ---
        draft_task_instructions = ""
        critique_criteria = ""
        refinement_instructions = ""
        final_doc_type_name = generation_type # Default

        # --- UPDATED PROMPTS FOR RESUME ---
        if generation_type == TYPE_RESUME:
            final_doc_type_name = "ATS-Optimized Resume (Selected Highlights)" # Updated name slightly
            draft_task_instructions = f"""
            **Role:** Expert Resume Writer & ATS Optimization Specialist adapting content to a precise format.
            **Goal:** Generate a **FIRST DRAFT** ATS-Optimized Resume in **Markdown format** using the **EXACT** formatting examples provided below. The resume MUST be easily parseable, compelling, incorporate keywords, AND **prioritize including the content from 'Selected Experience Highlights' and 'Selected Project Highlights' provided in the context below.**

            **Primary Objectives:**
            1.  **Prioritize Selected Highlights:** When generating the `## Experience` and `## Projects` sections, **ensure the content listed under `Selected Experience Highlights` and `Selected Project Highlights` (if provided below) is included and integrated naturally.** These represent the most relevant points.
            2.  **Precise Formatting:** Adhere STRICTLY to the Markdown formatting examples given for each section (H2 headings, Contact Info, Skills, Experience/Project structure with newlines).
            3.  **High ATS Score:** Ensure the structure remains parseable. Use only standard Markdown elements as specified.
            4.  **Keyword Optimization:** Naturally integrate  approximately 90% of 'Ranked Job Requirements' keywords throughout the resume (Summary, Skills, Experience bullets, Project bullets). Distribute contextually.
            5.  **Human Readability & Impact:** Maintain clarity, professionalism (`{tone}`), use strong action verbs,and quantify achievements.

            **CRITICAL ATS OPTIMIZATION & CONTENT RULES (Follow Strictly with Examples):**
            1.  **Standard Section Headings (H2):** Use **EXACTLY** `##` (Markdown H2) for: `## Summary`, `## Skills`, `## Experience`, `## Projects`, `## Certifications`. Optional standard sections (`## Education`, `## Awards`) also use `##`.
            2.  **Contact Information (Top, Left-Aligned - NO CENTERING):** Place contact info at VERY TOP, left-aligned. Use this precise format:
                ```markdown
                # {name}
                Email: {email} | LinkedIn: [Your LinkedIn URL] | [Your City, State]
                ```
                *(Use candidate's actual info. Keep it simple.)*
            3.  **Keyword Integration & Distribution Strategy:** Target ~80% coverage, distribute widely (Summary, Skills, Experience/Project bullets), integrate naturally.
            4.  **Targeted Summary:** `## Summary` MUST be **STRICTLY 1-2 sentences MAXIMUM**. Target to `{jd_title}`, include top keywords, make it a hook.
            5.  **Skills Section Format (MANDATORY EXAMPLE):** Format `## Skills` **EXACTLY** like examples. `* ` bullet, bold category, colon, space, comma-separated skills.
                ```markdown
                ## Skills
                * **Programming Languages:** Python (Pandas, NumPy, PySpark), SQL, R
                * **Cloud & Data Engineering Platforms:** Azure (Data Factory, Databricks, Synapse, ADLS Gen2, Blob Storage), AWS (Glue, EMR, S3, Redshift, Kinesis, Lambda), Snowflake, Databricks Delta Lake, Airflow, Airbyte, Docker, Kubernetes, dbt
                * **[Other Category]:** Skill A, Skill B, Skill C
                ```
            6.  **Experience Section Format (MANDATORY EXAMPLE + NEWLINE):** For EACH entry under `## Experience`:
                * First line: Job Header, formatted **EXACTLY** like this (bold title/co/loc, pipe separators, dates NOT bolded at end):
                  `**Data Analyst, Fintech | Open Financial Technology Pvt. Ltd | Bengaluru, Karnataka, India** (Mar 2022 – Aug 2023)`
                * **CRITICAL:** There **MUST** be a **newline character** immediately after the entire Job Header line (including the dates) and **BEFORE** the first `*` bullet point below it.
                * Subsequent lines: Accomplishments as **standard Markdown bullet points (`* `)** starting with a strong action verb. **Prioritize incorporating content from 'Selected Experience Highlights' here.** Aim for relevant bullets per job, using the highlights as the main source.
                * Follow this multi-line structure **EXACTLY**:
                ```markdown
                ## Experience

                **Data Analyst, Fintech | Open Financial Technology Pvt. Ltd | Bengaluru, Karnataka, India** (Mar 2022 – Aug 2023)

                * [Bullet incorporating selected highlight 1...]
                * [Bullet incorporating selected highlight 2...]
                * [Another relevant accomplishment, possibly derived from highlights or context...]

                **[Previous Job Title] | [Previous Company] | [Previous Location]** ([Start Date] – [End Date])

                * [Bullet incorporating selected highlight 3...]
                * [...]
                ```
            7.  **Projects Section Format (MANDATORY EXAMPLE + NEWLINE):** For EACH entry under `## Projects`:
                * First line: Project Header, formatted **EXACTLY** like this (bold project name only):
                  `**COVID-19 Data Pipeline and Analysis with Azure Data Factory**`
                * **CRITICAL:** There **MUST** be a **newline character** immediately after the Project Header line and **BEFORE** the first `*` bullet point below it.
                * Subsequent lines: Accomplishments/details as **standard Markdown bullet points (`* `)**. **Use the content provided in 'Selected Project Highlights' to construct these entries.**
                * Follow this multi-line structure **EXACTLY**:
                ```markdown
                ## Projects

                **[Project Name from Selected Highlight 1]**

                * [Detail from selected highlight 1...]
                * [Another detail from selected highlight 1...]

                **[Project Name from Selected Highlight 2]**

                * [Detail from selected highlight 2...]
                ```
            8.  **Certifications Section Format (MANDATORY EXAMPLE):** If `## Certifications` section used, list ALL certs as a **single, comma-separated string** below heading. **NO BULLETS.**
                ```markdown
                ## Certifications
                Azure Data Engineer Associate, Azure Data Fundamentals, IBM Data Science Professional Certificate, Tableau Desktop Specialist
                ```
            9.  **ATS-Friendly Formatting ONLY:** Use `* ` bullets ONLY as shown. AVOID tables, columns, images, icons, unusual symbols, HTML. Use standard dates.
            10. **Content Quality & Conciseness:** Be clear, professional (`{tone}`). Start bullets with action verbs. Quantify results. AVOID redundant statements. Focus on impact.

            **Task:** Generate the ATS-Optimized Resume draft in Markdown, adhering **STRICTLY** to ALL rules and **EXACT FORMATTING EXAMPLES**, **especially prioritizing the selected highlights** provided in the context for Experience and Projects sections. Ensure 1-2 sentence summary limit and ~80% keyword coverage/distribution. Output only Markdown.
            """
            # Critique criteria remains mostly the same, but we can add checks for highlight usage
            critique_criteria = f"""
            **ATS & Content Critique Criteria (Output brief bullet points ONLY, check against examples):**
            1.  **Overall Structure & Formatting Compliance:**
                * **Section Headings:** Uses **EXACTLY** `## Summary`, `## Skills`, etc.? (Yes/No/Incorrect Heading)
                * **Contact Info Format:** Top, left-aligned, matches `# Name \\n Contact Line` format? **NO CENTERING?** (Yes/No/Incorrect Format)
                * **Skills Section Format:** Adheres **EXACTLY** to `* **Category:** Skill1, ...` format per example? (Yes/No/Incorrect Format)
                * **Experience Section Format:** Headers match **EXACTLY** `**Title | Co | Loc** (Dates)`? Accomplishments use `* ` bullets below? **Crucially, is there a NEWLINE between the header line and the first bullet?** (Yes/No/Incorrect Header/Incorrect Bullets/Missing Newline)
                * **Projects Section Format:** Headers match **EXACTLY** `**Project Name**`? Accomplishments use `* ` bullets below? **Crucially, is there a NEWLINE between the header line and the first bullet?** (Yes/No/Incorrect Header/Incorrect Bullets/Missing Newline)
                * **Certifications Format (If Present):** If `## Certifications` exists, is content a single comma-separated string below heading (NO bullets)? (Yes/No/Incorrect Format/NA)
                * **Simple Formatting:** Consistent standard dates? **Absence** of tables, columns, complex symbols, HTML? (Yes/No/Issues Found)
            2.  **Keyword Optimization Assessment:**
                * **Keyword Coverage (% Estimate):** Appears to incorporate target ~80% of 'Ranked Job Requirements' keywords? (Estimate: e.g., "High ~80-90%", "Moderate ~60-70%", "Low <50%")
                * **Keyword Distribution:** Keywords distributed naturally across Summary, Skills, Experience/Project bullets? Or concentrated/sparse? (e.g., "Well-distributed", "Concentrated", "Sparse")
                * **Natural Integration:** Keywords integrated smoothly? Or feels stuffed? (e.g., "Natural", "Forced", "Stuffed")
            3.  **Content Quality & Clarity:**
                * **Summary Length:** Is `## Summary` **STRICTLY 1 or 2 sentences**? (Yes/No + Count)
                * **Summary Content:** Concise, tailored, impactful, includes top keywords within limit? (Yes/No/Needs Improvement)
                * **Experience/Project Bullets:** Start with strong action verbs? Quantified results present? Impact clear? Skills demonstrated? (Strong/Moderate/Weak)
                * **Selected Highlight Usage:** Does the generated Experience/Projects section **clearly incorporate content from the provided 'Selected Highlights'**? (Yes/Partially/No/NA) # <-- NEW CHECK
                * **Redundant Statements:** Avoids obvious statements like "* Meets degree requirement."? (Yes/No - Redundant statements found)
                * **Overall Readability & Tone:** Clear, concise, professional (`{tone}`), error-free? (Good/Fair/Poor)
            4.  **Requirement Alignment:** Content (esp. Experience/Project bullets) strongly evidences suitability for *top-ranked* job requirements? (Strong/Moderate/Weak)
            """
            refinement_instructions = f"""
            * **Address ALL Critique Points:** Focus meticulously on fixing any deviations from the **EXACT FORMATTING EXAMPLES**, **especially the required NEWLINES after Experience/Project headers**. Also fix keyword issues, summary length, content weaknesses, and remove redundant statements.
            * **Improve Highlight Incorporation:** If critique indicates poor usage of 'Selected Highlights', revise the Experience and Project sections to better feature that prioritized content. # <-- NEW INSTRUCTION
            * **Fix Formatting to Match Examples:** Correct Contact Info format (NO CENTERING). Ensure Skills format is `* **Category:** ...`. Ensure Experience headers are `**Title | Co | Loc** (Dates)` with a **NEWLINE** then `* ` bullets below. Ensure Project headers are `**Project Name**` with a **NEWLINE** then `* ` bullets below. Ensure Certifications is a comma-separated string (NO bullets). Remove ALL other ATS-unfriendly formatting.
            * **Optimize Keyword Coverage & Distribution:** If coverage < ~80% or distribution poor, revise Summary, Skills, and Experience/Project bullets to naturally weave in more relevant JD keywords. Ensure spread and context. Avoid stuffing.
            * **MANDATORY Summary Length Correction:** If critique found `## Summary` > 2 sentences, **MUST shorten to exactly 1-2 sentences**. Be ruthless. Retain essential hook/keywords.
            * **Improve Content Quality:** Strengthen action verbs in bullets. Add quantification. Enhance clarity. Improve skill demonstration within bullets. **Remove any identified redundant statements.**
            * **Final Polish:** Ensure coherence, professionalism (`{tone}`), zero errors. Output clean Markdown adhering strictly to all rules and formatting examples, including required newlines.
            """

# --- End of REPLACEMENT Block ---

        # ==================================================
        # --- PROMPTS FOR COVER LETTER (Deep Company Alignment) ---
        # ==================================================
        elif generation_type == TYPE_COVER_LETTER:
            # (Keep the existing Cover Letter prompts - no changes requested here)
            final_doc_type_name = "Cover Letter"
            draft_task_instructions = f"""
            **Role:** Expert Cover Letter Strategist crafting compelling narratives showing deep candidate-company alignment.
            **Goal:** Generate a persuasive **FIRST DRAFT** cover letter (approx. 300-450 words) for `{name}` applying to `{jd_title}` at `{jd_company}`. The letter must demonstrate genuine interest based on specific research (if available via `External Company Research Findings`) or JD context (`Job Description Company Context`) and connect the candidate's value directly to the company's needs and goals.

            **Instructions for Cover Letter Draft:**
            1.  **Hook:** Start with a strong, engaging opening paragraph. Immediately state the position (`{jd_title}`) being applied for. Connect `{name}`'s core value proposition or a key achievement directly to a primary need implied by the top 'Ranked Job Requirements'.
            2.  **Demonstrate Interest & Alignment (CRITICAL):**
                * Explicitly mention `{jd_company}`.
                * Show genuine, specific interest. Reference **at least one specific detail** about the company, ideally from `External Company Research Findings` (if available and not an error message) or otherwise from the `Job Description Company Context` (`{company_context_jd}`). Examples: recent news, a stated value, a mission aspect, a specific challenge mentioned.
                * **Crucially:** Explain *why* this specific detail resonates with the candidate or how their background/skills connect to it. Avoid generic praise. Example: "I was particularly drawn to [Specific Detail Found/Mentioned] because my experience in [Candidate Skill/Achievement] directly addresses the need for [Related Company Goal/Challenge]."
            3.  **Evidence-Based Body Paragraphs:** Select 1-2 *most relevant* achievements or experiences from the resume (`Parsed Candidate Resume Sections`, `Extracted Candidate Resume Achievements`) that strongly address the **top ranked requirements** (`{ranked_req_str}`). Quantify results whenever possible (use achievement data). Explicitly link how these qualifications will deliver value or solve problems for `{jd_company}` in the context of this role.
            4.  **Keyword Integration:** Naturally incorporate relevant keywords from the 'Ranked Job Requirements', including potentially some from 'Potentially Missing Keywords' if authentically supported by the candidate's experience. Do NOT force keywords unnaturally or list them. Flag uncertainty if necessary: `[Note: Assumed relevance based on X]`.
            5.  **Tone:** Maintain the specified primary '{tone}'. Ensure it is professional, enthusiastic, and confident throughout.
            6.  **Structure & Length:** Follow standard Cover Letter format (Introduction, 1-3 Body Paragraphs, Conclusion). Aim for approximately 300-450 words. Ensure logical flow between paragraphs.
            7.  **Conclusion:** Reiterate strong enthusiasm for the role and `{jd_company}`. Briefly summarize the core value proposition. Include a clear call to action, expressing eagerness for an interview to discuss qualifications further. Mention the attached resume.
            """
            critique_criteria = f"""
            **Critique Criteria (Output brief bullet points ONLY):**
            1.  **Alignment & Specificity:**
                * How well does the draft connect the candidate's skills/experience to the *specific* needs of the `{jd_title}` role at `{jd_company}`? (Scale: Strong/Moderate/Weak)
                * Does it use specific details effectively (from external research if provided, or JD context `{company_context_jd}`)? Is the expressed interest genuine and specific, or generic? (Scale: Specific/Somewhat Generic/Generic)
            2.  **Requirement Addressed & Evidence:**
                * Does it clearly address the *top-ranked* requirements (`{ranked_req_str}`)? (Yes/Partially/No)
                * Is the supporting evidence from the resume/achievements specific and compelling? Is quantification used effectively? (Yes/Somewhat/No)
                * Is the link between qualifications and providing value to the company clear? (Clear/Vague/Missing)
            3.  **Keyword Integration:** Are relevant keywords used naturally and contextually within sentences? (Natural/Forced/Sparse) Are `[Note: ...]` flags used appropriately if needed? (Yes/No/NA)
            4.  **Structure, Flow, & Length:** Is it well-organized (Intro/Body/Conclusion)? Does it flow logically? Is it within the target word count (approx. 300-450 words)? (Good/Fair/Poor)
            5.  **Clarity, Conciseness, & Tone:** Readability? Professionalism? Is the '{tone}' consistent and appropriate? Is the call to action clear and strong? (Good/Fair/Poor)
            6.  **Overall Impact & Persuasiveness:** Does the letter effectively sell the candidate and make a compelling case for an interview? (Strong/Moderate/Weak)
            """
            refinement_instructions = f"""
            * **Address Critique Directly:** Focus on fixing weaknesses identified in the critique, especially regarding specific company/role alignment, depth of interest, clarity of evidence, and overall impact.
            * **Strengthen Alignment & Specificity:** If alignment was weak/generic, find more concrete ways to link the candidate's strongest qualifications (`Parsed Candidate Resume Sections`, `Extracted Candidate Resume Achievements`) to the specific requirements (`{ranked_req_str}`) and company context (`{company_context_jd}` or external info). Make the 'why this company and this role' argument explicit and convincing. Remove/justify `[Note: ...]` flags.
            * **Enhance Evidence & Value Proposition:** Ensure the strongest, most relevant, and quantified achievements are used to back up claims about meeting top requirements. Clearly articulate the *impact* or value the candidate brings to `{jd_company}`.
            * **Refine Language & Flow:** Improve keyword integration naturally. Use strong action verbs. Enhance clarity, conciseness, and ensure the final '{tone}' is professional, confident, and engaging. Improve transitions between paragraphs.
            * **Format, Length & CTA:** Adhere to standard Cover Letter format, targeting approx. 300-450 words. Ensure a powerful closing paragraph and a clear, confident call to action.
            """
        else:
            # Handle unsupported generation types
            yield f"\n--- ERROR: Unsupported generation type '{generation_type}'. Cannot proceed. ---\n"
            return


        # --- Construct Full Prompts ---
        # Base context shared by draft and refinement prompts
        # ADD the selected highlights context conditionally for resumes
        base_context = f"""
        **Candidate:** {name} ({email})
        **Target Job:** {jd_title} at {jd_company}
        **Ranked Job Requirements (Most Important First):**
{ranked_req_str}
        **Parsed Candidate Resume Sections (Use for context, but prioritize Selected Highlights below for Exp/Proj):**
{resume_sections_str}
        **Extracted Candidate Resume Achievements (Sample - General Context):**
{achievements_sample_str}
        {selected_bullets_context_str if generation_type == TYPE_RESUME and selected_bullets_context_str else ""}
        {selected_projects_context_str if generation_type == TYPE_RESUME and selected_projects_context_str else ""}
        **Desired Tone:** {tone}
        **Potentially Missing Keywords (Resume vs. JD):** {missing_keywords_str}
        **Job Description Company Context:** {company_context_jd}
        {external_company_prompt_section if generation_type == TYPE_COVER_LETTER else ""}
        """ # Include external info only for Cover Letter context (if available)

        # Prompt for generating the initial draft
        draft_prompt = f"""
        {base_context}

        {draft_task_instructions}

        **Output ONLY the raw {final_doc_type_name} draft content. Do NOT include any introductory text like "Here is the draft..." or any explanations before or after the document content.**
        """

        # --- Multi-Turn Generation Process ---

        # --- Step 3: Generate Initial Draft ---
        yield f"--- Generating initial {final_doc_type_name} draft... ---\n"
        initial_draft = ""
        try:
            # Use the helper function for the LLM call
            initial_draft_result = await _call_llm_async(
                prompt=draft_prompt,
                api_key=google_api_key,
                model_name=GENERATION_MODEL_NAME,
                temperature=0.6 # Adjust temp as needed
            )
            if isinstance(initial_draft_result, str) and initial_draft_result.strip():
                initial_draft = initial_draft_result.strip()
            else:
                raise ValueError("Initial draft generation failed or returned empty.")
        except Exception as draft_err:
            yield f"\n--- ERROR: Initial {final_doc_type_name} draft generation failed: {draft_err} ---\n"
            logging.error(f"Initial {final_doc_type_name} draft generation failed.", exc_info=True)
            yield "\n--- Generation stopped due to draft failure. ---\n"
            return # Stop generation


        # --- Step 4: Generate Critique ---
        yield f"\n--- Generating critique for {final_doc_type_name}... ---\n"
        critique = "Critique generation failed or skipped." # Default
        # Construct critique prompt using the generated initial draft
        critique_prompt = f"""
        **Critique Task:** Evaluate the initial draft below based *strictly* on the critique criteria provided. Output only brief, specific bullet points addressing each criterion. Do not add explanations.

        **Context for Critique:**
        * **Target Job Requirements (Ranked):**
{ranked_req_str}
        * **Potentially Missing Keywords Attempted:** {missing_keywords_str}
        * **Desired Tone:** {tone}
        {f'* **JD Company Context:** {company_context_jd}' if generation_type == TYPE_COVER_LETTER else ''}
        {f'* **Selected Experience Highlights:** {selected_bullets_context_str}' if generation_type == TYPE_RESUME and selected_bullets_context_str else ''}
        {f'* **Selected Project Highlights:** {selected_projects_context_str}' if generation_type == TYPE_RESUME and selected_projects_context_str else ''}
        {external_company_prompt_section if generation_type == TYPE_COVER_LETTER else ""}

        **Initial {final_doc_type_name} Draft to Critique:**
        ---
        {initial_draft}
        ---

        **Critique Criteria to Address (Output Bullets ONLY):**
        {critique_criteria}
        """ # Added selected context to critique prompt
        try:
            # Use helper function for critique LLM call
            critique_result = await _call_llm_async(
                prompt=critique_prompt,
                api_key=google_api_key,
                model_name=EXTRACTION_MODEL_NAME, # Use extraction model
                temperature=0.15 # Low temp
            )
            if isinstance(critique_result, str) and critique_result.strip():
                critique = critique_result.strip()
                yield critique + "\n" # Show critique in stream
            else:
                logging.warning(f"Critique generation for {final_doc_type_name} returned empty or non-string. Proceeding without specific critique.")
                yield f"\n--- WARNING: Critique generation failed or was empty. Attempting refinement based on initial draft and instructions only. ---\n"
        except Exception as critique_err:
            yield f"\n--- ERROR: Critique generation failed: {critique_err} ---\n"
            logging.error(f"Critique generation for {final_doc_type_name} failed.", exc_info=True)
            yield f"\n--- Proceeding with refinement based on initial draft and instructions only. ---\n"


        # --- Step 5: Generate Final Refined Version (Streaming) ---
        yield f"\n--- Generating final refined {final_doc_type_name}... ---\n"
        # Construct refinement prompt using initial draft and critique
        refinement_prompt = f"""
        **Refinement Task:** Generate the **FINAL, REVISED** {final_doc_type_name}. Your goal is to meticulously address the critique points provided below (if any) AND strictly adhere to ALL original instructions and rules outlined in the base context (including prioritizing selected highlights for Resumes). Produce a polished, complete final document.

        {base_context} # Includes selected highlights if applicable

        **Initial {final_doc_type_name} Draft:**
        ---
        {initial_draft}
        ---
        **Critique of Initial Draft (Address these points):**
        ---
        {critique}
        ---

        **Revision Instructions (Apply these based on critique and original goals):**
        {refinement_instructions}

        **Output ONLY the final revised {final_doc_type_name}. Do not include any extra commentary, introductions, or explanations before or after the document content. Just output the complete, revised document.**
        """
        try:
            # Use helper function for final generation stream
            final_stream_generator = await _call_llm_async(
                prompt=refinement_prompt,
                api_key=google_api_key,
                model_name=GENERATION_MODEL_NAME, # Use generation model
                temperature=0.5, # Adjust temp as needed
                stream=True
            )

            # Stream the final output chunk by chunk
            stream_produced_output = False
            if hasattr(final_stream_generator, '__aiter__'):
                async for chunk in final_stream_generator:
                    # Ensure chunk is string before yielding
                    if isinstance(chunk, str):
                        yield chunk
                        stream_produced_output = True
                    else:
                        logging.warning(f"Received non-string chunk in final stream: {type(chunk)}")

                if not stream_produced_output:
                    yield f"\n--- WARNING: Final {final_doc_type_name} generation stream finished without producing output. Check for potential prompt/model issues. ---"
                    logging.warning(f"Final {final_doc_type_name} stream was empty.")
            else:
                # Handle cases where streaming call unexpectedly didn't return a generator
                error_message = f"\n--- ERROR: Expected stream, received unexpected type ({type(final_stream_generator)}) during final {final_doc_type_name} generation. ---"
                logging.error(error_message + f" Content (if any): {str(final_stream_generator)[:200]}")
                yield error_message

        except Exception as final_gen_err:
            yield f"\n--- ERROR: Final {final_doc_type_name} generation failed during streaming: {final_gen_err} ---\n"
            logging.error(f"Final {final_doc_type_name} generation/streaming failed.", exc_info=True)
            yield "\n--- Generation stopped due to final refinement failure. ---\n"
            return


    # --- Global Exception Handler for the entire function ---
    except Exception as e:
        error_message = f"\n--- Unexpected Critical Error during {generation_type} Generation Pipeline: {e} ---"
        logging.error(f"Critical error in generate_application_text_streamed for {generation_type}: {e}", exc_info=True)
        yield error_message
        # Provide available context if helpful for debugging
        if common_data:
            yield f"\nDebug Info (Preparation Stage Error): {common_data.get('error', 'None')}"
            yield f"\nDebug Info (JD Data Keys): {list(common_data.get('jd_data', {}).keys())}"
            yield f"\nDebug Info (Resume Data Keys): {list(common_data.get('resume_data', {}).keys())}"
        yield "\n--- Generation stopped due to critical pipeline error. ---"


# --- Email Generator + Validator (Async) ---
async def generate_email_and_validate(
    name: str, email: str, job_description: str, resume_content: str,
    google_api_key: str, tone: str, email_recipient_type: str,
    job_link: Optional[str] = None
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Generates an impactful job application email, incorporates job link, and validates ATS friendliness (async)."""
    generated_email: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None
    common_data: Dict[str, Any] = {} # Store data prep results

    try:
        # --- 1. Prepare Common Data ---
        # Use the same robust data prep function
        logging.info("Email Gen: Preparing common data...")
        common_data = await _prepare_common_data(job_description, resume_content, google_api_key)

        # --- Handle Preparation Errors ---
        if common_data.get("error"):
            error_msg = f"Email generation stopped during data preparation: {common_data['error']}"
            logging.error(error_msg)
            # Return None for email, and error info in validation dict
            return None, {"error": f"Data Preparation Error: {common_data['error']}", "ats_score": None, "detailed_checks": {}, "overall_feedback": error_msg}

        # Ensure critical JD info needed for the email is present
        jd_data = common_data.get("jd_data", {})
        jd_title = jd_data.get("job_title")
        jd_company = jd_data.get("company_name")
        if not jd_title or not jd_company:
            error_msg = "Email generation stopped: Missing critical JD data (Job Title or Company Name) after preparation."
            logging.error(error_msg)
            return None, {"error": error_msg, "ats_score": None, "detailed_checks": {}, "overall_feedback": error_msg}


        # --- 2. Format Inputs for Email Prompt ---
        logging.info("Email Gen: Formatting inputs for prompt...")
        # Extract relevant pieces from common_data safely
        resume_data = common_data.get("resume_data", {})
        resume_achievements = resume_data.get("achievements", [])
        ranked_reqs = common_data.get("ranked_jd_requirements", [])
        company_context = jd_data.get('company_values_mission_challenges', '')

        # Prepare a concise summary/highlight for the email prompt
        resume_summary_for_prompt = "Relevant skills and experience outlined in attached resume." # Default
        summary_section = resume_data.get("sections", {}).get("summary")
        if summary_section and isinstance(summary_section, str) and len(summary_section) < 400:
             resume_summary_for_prompt = f"Candidate Summary Snippet: {summary_section}"
        # Alternative: Use top keywords
        # top_keywords = resume_data.get("extracted_keywords", [])[:5]
        # if top_keywords: resume_summary_for_prompt = f"Key Skills Include: {', '.join(top_keywords)}"

        # Focus on Top 3 requirements for the email's context
        top_3_reqs_str = "\n".join(f"- {req}" for req in ranked_reqs[:3]) if ranked_reqs else "N/A"

        # Select ONE strong, preferably quantified, achievement for the prompt context
        strongest_achievement_str = "N/A"
        # Prioritize quantified achievements
        quantified_achievements = [a for a in resume_achievements if a.get("quantifiable_result")]
        if quantified_achievements:
            # Maybe pick the one with the most impressive number? Simple approach: first one.
            ach = quantified_achievements[0]
            strongest_achievement_str = f"{ach['action_verb']}... (Result: {ach['quantifiable_result']})"
        elif resume_achievements: # Fallback to the first achievement if no quantified ones
             ach = resume_achievements[0]
             strongest_achievement_str = f"{ach['action_verb']}..." # No result mentioned

        # Define recipient focus based on type
        recipient_focus_desc = ""
        if email_recipient_type == RECIPIENT_TA_HM:
            recipient_focus_desc = "Tailor message for Talent Acquisition/Hiring Manager. Focus on direct alignment with top requirements, quantifiable results, ROI potential, problem-solving ability, and clear value proposition. Be concise and professional."
        else: # RECIPIENT_GENERAL
            recipient_focus_desc = "Tailor message for a general application or unspecified recipient. Focus on clear communication, enthusiasm, professionalism, highlighting transferable skills, and key qualifications relevant to the role and company. Ensure easy readability and clarity."

        # Prepare job link display for prompt
        job_link_display = job_link if job_link and job_link.strip().startswith(('http:', 'https:')) else "Not Provided"


        # --- 3. Construct Email Generation Prompt ---
        email_prompt = f"""
        **Role:** Expert Email Copywriter specializing in concise, impactful, and professional job application emails.
        **Goal:** Generate a compelling email (target word count: 150-250 words) for `{name}` to express interest in the `{jd_title}` role at `{jd_company}`. The email should be tailored to the recipient type and encourage them to review the application.

        **Candidate Info:**
        * Name: {name}
        * Email: {email}

        **Target Information:**
        * Job Title: {jd_title}
        * Company: {jd_company}
        * Job Posting Link: {job_link_display}
        * Recipient Type: {email_recipient_type}
        * Recipient Focus: {recipient_focus_desc}
        * Desired Tone: {tone} (apply professionally, ensure confidence and impact)

        **Key Context for Email Content:**
        * Top 3 Job Requirements (Ranked):
            {top_3_reqs_str}
        * Example Candidate Achievement: {strongest_achievement_str}
        * Brief Company Context (from JD): {company_context or 'N/A'}
        * Candidate Resume Highlight: {resume_summary_for_prompt}

        **Instructions for Email Generation:**
        1.  **Subject Line:** Create a clear, professional, and informative subject line. Include the Job Title and Candidate Name (e.g., `Application: {jd_title} - {name}` or `Interest in {jd_title} Opportunity - {name}`).
        2.  **Opening:** Start concisely. State the purpose (applying for `{jd_title}`) and immediately connect the candidate's core strength or the example achievement to a top requirement.
        3.  **Body Paragraph 1 (Evidence & Impact):** Briefly elaborate on the candidate's suitability. Showcase the single most impactful relevant skill or achievement (drawing inspiration from the example achievement provided). If possible, hint at the quantifiable impact. Tailor the emphasis based on the `{recipient_focus_desc}`.
        4.  **Body Paragraph 2 (Company Connection - Brief):** Briefly express genuine interest in `{jd_company}` or the specific role. If the `{company_context}` provides useful detail, reference it concisely to show alignment (e.g., "I'm particularly interested in [Company]'s work in [area from context]..."). Keep this very short (1-2 sentences max).
        5.  **Keyword Integration:** Naturally weave in 1-2 other relevant keywords from the top requirements list if it fits the flow. Prioritize clarity and natural language over forcing keywords.
        6.  **Job Link Reference:** If a valid Job Posting Link was provided (`{job_link_display}` is a URL), smoothly mention applying via that link or reference the specific posting (e.g., "...applying for the {jd_title} position posted [here/on LinkedIn/your website]."). Omit reference if no link was provided.
        7.  **Conciseness & Tone:** Adhere strictly to the '{tone}'. Be highly concise (aim for 150-250 words). Use clear, professional language appropriate for the recipient. Ensure excellent readability.
        8.  **Call to Action:** Conclude professionally. Express strong enthusiasm for the opportunity and indicate readiness for next steps (e.g., "I am eager to discuss how my skills can benefit [Company]..."). Mention that the resume is attached or was submitted via the link/portal.
        9.  **Signature:** Include Name and Email. Optionally include Phone and LinkedIn profile URL.

        **Output ONLY the email content, starting with the Subject line and followed by the body and signature. No extra commentary or explanations.**

        Subject: [Your Subject Line Here]

        [Body of the email, approx. 150-250 words]

        [Signature Block:
        {name}
        {email}
        [Optional: Phone Number]
        [Optional: LinkedIn URL]
        ]
        """

        # --- 4. Generate Email ---
        logging.info("Email Gen: Calling LLM to generate email...")
        generated_email_result = await _call_llm_async(
            prompt=email_prompt,
            api_key=google_api_key,
            model_name=GENERATION_MODEL_NAME,
            temperature=0.65 # Temp balanced for professionalism and slight creativity
        )

        if not isinstance(generated_email_result, str) or not generated_email_result.strip():
             # Log error and return failure
             error_msg = "Email generation call failed or returned empty content."
             logging.error(error_msg)
             prep_error = common_data.get("error", "None")
             return None, {"error": error_msg, "debug_prep_error": prep_error, "ats_score": None, "detailed_checks": {}, "overall_feedback": error_msg}

        generated_email = generated_email_result.strip()
        logging.info("Email Gen: Email content generated successfully.")


        # --- 5. Validate Generated Email ---
        logging.info("Email Gen: Validating generated email...")
        # Pass the generated email and necessary context to the validator
        validation_results = await _validate_ats_friendliness(
            document_text=generated_email,
            document_type=TYPE_EMAIL, # Use constant
            job_description_data=common_data.get("jd_data", {}), # Pass full structured JD data
            google_api_key=google_api_key
        )

        # Log validation outcome
        if validation_results.get("error"):
            logging.warning(f"Email validation encountered an error: {validation_results['error']}")
        else:
            logging.info(f"Email validation completed. ATS Score Impression: {validation_results.get('ats_score', 'N/A')}")


        # --- Return Results ---
        return generated_email, validation_results

    # --- Handle Critical Errors in the Pipeline ---
    except Exception as e:
        critical_error_msg = f"Critical error in generate_email_and_validate pipeline: {e}"
        logging.error(critical_error_msg, exc_info=True)
        # Include prep error info if available and seems relevant
        prep_error_info = ""
        if common_data and common_data.get("error"):
             prep_error_info = f" (Data Prep Error: {common_data['error']})"
        final_error_msg = f"Email Generation/Validation Pipeline Error: {e}{prep_error_info}"
        # Return None for email, and error info in validation dict
        return None, {"error": final_error_msg, "ats_score": None, "detailed_checks": {}, "overall_feedback": final_error_msg}



# --- ATS Validator (Async - Enhanced Checks) ---
async def _validate_ats_friendliness(
    document_text: str, document_type: str, job_description_data: Dict, google_api_key: str
) -> Dict[str, Any]:
    """Uses an LLM to evaluate ATS friendliness and content relevance with granular checks (async)."""

    # --- Input Validation ---
    if not document_text or not document_text.strip():
        logging.warning("No document text provided for ATS validation.")
        return {"error": "No document text provided for validation.", "ats_score": None, "detailed_checks": {}, "overall_feedback": "No document text."}
    if not isinstance(job_description_data, dict):
         logging.warning(f"Invalid job_description_data type ({type(job_description_data)}) for ATS validation.")
         # Proceed with validation but keyword checks will be limited
         job_description_data = {} # Use empty dict to avoid errors downstream


    # Initialize results structure
    results = {
        "error": None,
        "ats_score": None, # Placeholder for 1-5 score
        "keyword_check": {
            "found_keywords": [],
            "missing_suggestions": [],
            "density_impression": "N/A"
        },
        "clarity_structure_check": "N/A",
        "formatting_check": "N/A",
        "detailed_checks": {}, # Will be populated based on document_type
        "overall_feedback": "N/A",
        "raw_response": None # Store raw response for debugging if needed
    }

    try:
        # --- Prepare Context for Prompt ---
        # Safely extract JD keywords for context
        jd_keywords_list = job_description_data.get("key_skills_requirements", [])
        jd_keywords_str = "N/A (Requirements not available)"
        if isinstance(jd_keywords_list, list) and jd_keywords_list:
            # Join valid string representations of keywords
            jd_keywords_str = ", ".join(filter(None, map(str, jd_keywords_list)))
        elif isinstance(jd_keywords_list, list): # It's a list, but empty
             jd_keywords_str = "N/A (No requirements listed in JD data)"

        jd_title = job_description_data.get("job_title", "N/A")

        # --- Define Detailed Checks Specific to Document Type ---
        detailed_checks_instructions = "* (No specific detailed checks defined for this type)" # Default
        # Use constants for type checking for robustness
        if document_type == TYPE_RESUME:
            detailed_checks_instructions = """
            * `Standard Sections Present`: Check if standard headings (Summary, Skills, Experience, Education) seem present using `##` Markdown. (Rate: "Yes", "Partially", "No")
            * `Clear Date Formats`: Assess if experience/education dates use a consistent, standard, parsable format (e.g., Month Year – Month Year/Present). (Rate: "Good", "Fair", "Inconsistent/Missing")
            * `Action Verbs Used`: Does the Experience section predominantly start bullet points with strong action verbs? (Rate: "Strong", "Moderate", "Weak/Limited")
            * `Quantifiable Results Present`: Are there indicators of quantified results (using %, #, $) in the Experience section? (Rate: "Numerous", "Some", "Few/None")
            * `Contact Info Clear`: Is contact info (Name, Email, Phone, Location) clearly presented, usually at the top? (Rate: "Yes", "Partial/Ambiguous", "Missing")
            * `Skills Section Format Check`: Does the Skills section appear to follow the specified `* **Category:** ...` format? (Rate: "Yes", "Partially/Inconsistent", "No/Absent")
            * `Formatting Issues Check`: Any obvious ATS-unfriendly elements like tables, columns, images (implied), complex symbols, text boxes? (Rate: "None Visible", "Minor Potential Issues", "Major Issues Likely")
            * `Summary Length Check`: Does the `## Summary` section strictly contain 1 or 2 sentences? (Rate: "Yes (1-2 sentences)", "No (>2 sentences)", "Missing")
            """
        elif document_type == TYPE_COVER_LETTER:
            detailed_checks_instructions = """
            * `Standard CL Structure`: Follows a logical flow (Introduction, Body Paragraphs, Conclusion)? (Rate: "Clear Structure", "Somewhat Disorganized", "Poor Structure")
            * `Conciseness Check`: Is the letter length appropriate (e.g., typically under 1 page / ~300-500 words)? (Rate: "Good Length", "Potentially Too Long", "Too Brief")
            * `Clarity of Purpose`: Is the target role (`{jd_title}`) and company clearly stated early on? (Rate: "Clear", "Slightly Vague", "Unclear/Missing")
            * `Action Verbs Used`: Does the body effectively use action verbs when describing experiences/skills? (Rate: "Strong Use", "Moderate Use", "Limited Use")
            * `Quantifiable Results Present`: Are any specific metrics or quantified achievements mentioned to support claims? (Rate: "Yes", "Few/None")
            * `Call to Action`: Is there a clear and confident call to action in the conclusion? (Rate: "Clear & Strong", "Present but Weak", "Missing")
            """
        elif document_type == TYPE_EMAIL:
            detailed_checks_instructions = """
            * `Subject Line Clarity`: Is the subject line professional, clear, and informative (includes role/name)? (Rate: "Clear & Informative", "Okay but Generic", "Vague/Missing")
            * `Conciseness Check`: Is the email body appropriately brief and to the point (e.g., ~150-250 words)? (Rate: "Concise", "Slightly Long", "Too Brief/Abrupt")
            * `Clarity of Purpose`: Is the main reason for the email (applying for/inquiring about the role) immediately clear in the opening? (Rate: "Very Clear", "Moderately Clear", "Unclear")
            * `Call to Action`: Is there a clear statement about next steps or attached documents? (Rate: "Clear", "Implied/Weak", "Missing")
            * `Signature / Contact Info`: Is there a professional signature block with necessary contact info (Name, Email)? (Rate: "Complete", "Partial", "Missing")
            """

        # --- Construct Validation Prompt ---
        prompt = f"""
        **Task:** Perform a detailed ATS (Applicant Tracking System) friendliness and content relevance evaluation of the following '{document_type}' document, intended for the '{jd_title}' role.
        **Job Description Keywords/Requirements (for context only):** {jd_keywords_str}

        **Document Text to Evaluate:**
        ---
        {document_text}
        ---

        **Evaluation Criteria & Output Format:**
        Return ONLY a valid JSON object containing the following keys. Provide honest, critical assessments based *only* on the text provided and general ATS/recruiting best practices. Use the rating scales suggested in the detailed checks.

        1.  `ats_score` (integer, 1-5): Your overall assessment of the document's ATS compatibility (1=Very Poor, 3=Fair, 5=Excellent). Consider structure, formatting, keyword potential, and parseability.
        2.  `keyword_check` (object): Analyze keyword relevance against the JD context provided above. {{
                "found_keywords": [List of ~10-15 relevant keyword strings (lowercase) likely matching the JD context found IN THE DOCUMENT],
                "missing_suggestions": [List of ~5-10 important keywords from the JD context that seem ABSENT or underrepresented in the document],
                "density_impression": "string (Your qualitative impression, e.g., 'Good keyword density and natural integration', 'Keywords seem sparse', 'Keywords heavily concentrated in one section', 'Potential keyword stuffing')"
            }}
        3.  `clarity_structure_check` (string): Assess the overall clarity, organization, and logical flow for both ATS parsing and human readability (e.g., "Well-structured and easy to follow", "Moderately clear, some long paragraphs/sections", "Structure is confusing or illogical").
        4.  `formatting_check` (string): Evaluate the use of ATS-friendly formatting (standard fonts implied, standard bullets, consistent dates, no complex elements). (e.g., "Clean standard formatting", "Minor issues (e.g., inconsistent dates or non-standard bullets)", "Major issues likely (e.g., hints of tables, columns, unusual characters)").
        5.  `detailed_checks` (object): Provide brief, specific ratings for the document based *only* on its text, addressing the points listed below. Use the suggested rating scales within the parentheses. {{
                {detailed_checks_instructions}
            }}
        6.  `overall_feedback` (string): Provide brief (2-3 sentences), actionable feedback summarizing the document's key strengths and weaknesses regarding its purpose (ATS compatibility, convincing a recruiter, etc.). Highlight the most critical areas for improvement.

        **JSON Output:**
        """

        # --- Call LLM for Validation ---
        logging.info(f"ATS Validation: Calling LLM for {document_type}...")
        response_text = await _call_llm_async(
            prompt=prompt,
            api_key=google_api_key,
            model_name=EXTRACTION_MODEL_NAME, # Use model good for analysis/extraction
            temperature=0.1, # Very low temp for objective evaluation
            request_json=True # Request JSON output
        )

        # --- Process Validation Response ---
        if isinstance(response_text, str):
            raw_json_string = response_text # Assumed cleaned by helper
            results["raw_response"] = raw_json_string # Store raw response

            # --- Parse and Validate Validation JSON ---
            try:
                validation_data = json.loads(raw_json_string)

                if not isinstance(validation_data, dict):
                    raise ValueError("Validation response from LLM was not a dictionary.")

                # --- Populate results dict, validating types/structure ---
                required_keys = list(results.keys()) # Get keys from initialized dict (excl. raw_response)
                required_keys.remove("raw_response")
                required_keys.remove("error") # Error is handled separately

                # Check all required keys are present
                if not all(key in validation_data for key in required_keys):
                     missing = [key for key in required_keys if key not in validation_data]
                     raise ValueError(f"Validation response missing required keys: {missing}")

                # Validate ats_score
                ats_score = validation_data.get("ats_score")
                if isinstance(ats_score, int) and 1 <= ats_score <= 5:
                    results["ats_score"] = ats_score
                else:
                     logging.warning(f"Invalid ATS score received ({ats_score}). Setting to None.")
                     results["ats_score"] = None

                # Validate keyword_check structure and sanitize lists
                kw_check = validation_data.get("keyword_check")
                kw_required = ["found_keywords", "missing_suggestions", "density_impression"]
                if isinstance(kw_check, dict) and all(k in kw_check for k in kw_required):
                    results["keyword_check"]["found_keywords"] = sorted(list(set([
                        str(kw).lower().strip() for kw in kw_check.get("found_keywords", []) if isinstance(kw, str) and str(kw).strip()
                    ])))
                    results["keyword_check"]["missing_suggestions"] = sorted(list(set([
                        str(kw).lower().strip() for kw in kw_check.get("missing_suggestions", []) if isinstance(kw, str) and str(kw).strip()
                    ])))
                    results["keyword_check"]["density_impression"] = str(kw_check.get("density_impression", "N/A"))
                else:
                     logging.warning("Invalid keyword_check structure received.")
                     # Keep default empty/N/A values in results["keyword_check"]

                # Validate detailed_checks is a dict
                dt_check = validation_data.get("detailed_checks")
                if isinstance(dt_check, dict):
                    # Basic check: ensure values are strings or simple types if populated
                    results["detailed_checks"] = {k: str(v) for k, v in dt_check.items()}
                else:
                     logging.warning("Invalid detailed_checks structure received.")
                     results["detailed_checks"] = {} # Keep default empty dict

                # Validate other fields are strings
                results["clarity_structure_check"] = str(validation_data.get("clarity_structure_check", "N/A"))
                results["formatting_check"] = str(validation_data.get("formatting_check", "N/A"))
                results["overall_feedback"] = str(validation_data.get("overall_feedback", "N/A"))

                logging.info(f"Enhanced ATS Validation successful for {document_type}.")
                results["error"] = None # Explicitly confirm no error

            # --- Handle JSON Parsing/Validation Errors ---
            except (json.JSONDecodeError, ValueError) as json_err:
                error_msg = f"Failed to parse/validate LLM validation JSON response: {json_err}"
                logging.error(f"{error_msg}. Raw Response Snippet: {raw_json_string[:500]}...")
                results["error"] = error_msg
                # Reset other fields to default/None as validation failed
                results["ats_score"] = None
                results["detailed_checks"] = {}
                results["overall_feedback"] = "Validation failed."
            except Exception as parse_err:
                error_msg = f"Unexpected error processing LLM validation response: {parse_err}"
                logging.error(error_msg, exc_info=True)
                results["error"] = error_msg
                results["ats_score"] = None
                results["detailed_checks"] = {}
                results["overall_feedback"] = "Validation processing error."
        else:
             # Handle case where LLM call did not return a string
             logging.error(f"Unexpected return type from _call_llm_async for ATS validation: {type(response_text)}")
             results["error"] = "Error: LLM call for ATS validation did not return text."
             results["overall_feedback"] = "Validation LLM call failed."

    # --- Handle Outer LLM Call Errors ---
    except Exception as e:
        error_msg = f"Error during ATS validation LLM call for {document_type}: {e}"
        logging.error(error_msg, exc_info=True)
        results["error"] = f"ATS Validation LLM Call Error: {e}"
        results["overall_feedback"] = "Validation LLM call failed."

    return results