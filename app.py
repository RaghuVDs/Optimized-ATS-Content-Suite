# app.py

import streamlit as st
import llm_handler  # Import the updated module
import os
from io import StringIO
import time
import asyncio # Import asyncio
import re # For cleaning text and finding notes
import logging # Import logging
from typing import Optional, Dict, List, Any, AsyncGenerator
import textstat # Import textstat for readability
import nest_asyncio
nest_asyncio.apply()
import google.generativeai as genai

# --- Constants ---
# UPDATE THIS PATH if your default resume is located elsewhere
DEFAULT_RESUME_TXT_PATH = "default_resume.txt"

# --- Page Configuration ---
st.set_page_config(
    page_title="Hyper-Optimized ATS Suite Generator",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [App] %(message)s')

# --- Main Function ---
def main():

    # --- Helper Functions ---
    def get_api_key():
        """Retrieves the Google API Key from secrets or user input."""
        try:
            # Recommended: Use Streamlit secrets
            api_key = st.secrets["GOOGLE_API_KEY"]
            # st.sidebar.success("API Key loaded from Secrets.", icon="✅") # Keep sidebar clean
            return api_key
        except (FileNotFoundError, KeyError):
            # Fallback to user input if secrets aren't configured
            st.sidebar.warning("API Key not found in Secrets.", icon="⚠️")
            api_key = st.sidebar.text_input("Enter Google API Key:", type="password", key="api_key_input")
            if not api_key:
                st.sidebar.error("API Key is required.", icon="❌")
                return None
            st.sidebar.caption("Tip: Set up Streamlit Secrets for secure key management.")
            return api_key

    def read_text_file(file_path: str) -> Optional[str]:
        """Reads content from a text file, handling errors."""
        if not os.path.exists(file_path):
            st.sidebar.error(f"Default resume file not found: {os.path.basename(file_path)}", icon="❌")
            logging.error(f"Default resume file not found at: {file_path}")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                st.sidebar.warning(f"Default resume file '{os.path.basename(file_path)}' is empty.", icon="⚠️")
                logging.warning(f"Default resume file '{file_path}' is empty.")
                return None # Return None for empty file, handle later
            return content.strip() # Return content if not empty
        except Exception as e:
            st.sidebar.error(f"Error reading default resume: {e}", icon="❌")
            logging.error(f"Error reading default resume file '{file_path}': {e}", exc_info=True)
            return None

    async def _collect_stream(async_generator: AsyncGenerator[str, None]) -> Dict[str, Any]:
        """Helper to collect text from an async stream generator and handle errors."""
        full_text_list = []
        error_found = None
        try:
            async for chunk in async_generator:
                if isinstance(chunk, str):
                    # Basic check for embedded error markers from the generator
                    if chunk.strip().startswith("--- ERROR"):
                        error_found = chunk.strip()
                        logging.error(f"Error message yielded by stream: {error_found}")
                        # Collect text before breaking to provide context
                        break
                    full_text_list.append(chunk)
                else:
                    logging.warning(f"Received non-string chunk from stream: {type(chunk)}")

            # Check for error *after* iterating through the stream
            if error_found:
                collected_text = "".join(full_text_list)
                # Combine the error marker with any text collected before it stopped
                full_error_context = f"{error_found}\n(Collected text before error: '{collected_text[:200]}...')"
                return {"error": full_error_context}

            # Process successful stream collection
            full_text = "".join(full_text_list)
            # Clean up potential generation markers added in llm_handler
            full_text = re.sub(r'^---.*?---\n?', '', full_text, flags=re.MULTILINE)

            if not full_text.strip():
                logging.warning("Generation stream completed but yielded no text content.")
                return {"error": "Generation stream completed but yielded no text content."}

            return {"text": full_text.strip()}

        except Exception as e:
            logging.error(f"Error collecting stream: {e}", exc_info=True)
            return {"error": f"Stream collection error: {e}"}

    # --- Async Task Runner ---
    async def run_generation_tasks(tasks_to_run: Dict[str, asyncio.Task], prep_data: Dict, api_key: str):
        """Runs tasks, stores results, queues and runs validation tasks."""
        st.session_state.generation_results = {} # Reset results for this run
        st.session_state.validation_results = {} # Reset validation

        spinners = {}
        overall_spinner_placeholder = st.empty()
        overall_spinner_placeholder.info("🚀 Starting asynchronous generation tasks...")
        status_placeholder = st.container() # For individual statuses

        for name in tasks_to_run.keys():
            spinners[name] = status_placeholder.empty()
            spinners[name].info(f"⏳ Queued {name} generation...")

        # --- Gather Generation Results ---
        results = await asyncio.gather(*tasks_to_run.values(), return_exceptions=True)
        all_results = dict(zip(tasks_to_run.keys(), results))

        overall_spinner_placeholder.info("Processing generation results...") # Update overall status

        # --- Process Generation Results & Queue Validation ---
        validation_tasks = {}
        jd_data_for_validation = prep_data.get("jd_data", {}) # Use prepped JD data if available

        for name, result in all_results.items():
            spinners[name].empty() # Clear individual queued message
            status_placeholder.info(f"⚙️ Processing result for {name}...")

            if isinstance(result, Exception):
                error_msg = f"Task failed: {result}"
                st.session_state.generation_results[name] = {"error": error_msg}
                logging.error(f"Task {name} failed in gather: {result}", exc_info=result)

            elif name == "Email": # Tuple result from generate_email_and_validate
                if isinstance(result, tuple) and len(result) == 2:
                    email_text, validation_dict = result
                    if email_text:
                        st.session_state.generation_results[name] = {"text": email_text}
                        # Validation result is directly included
                        st.session_state.validation_results[name] = validation_dict or {"warning": "No validation data returned."}
                    else:
                        error_msg = validation_dict.get("error", "Email generation returned None.") if validation_dict else "Email gen failed."
                        st.session_state.generation_results[name] = {"error": error_msg}
                        if validation_dict: st.session_state.validation_results[name] = validation_dict
                else:
                    st.session_state.generation_results[name] = {"error": "Email task bad format."}

            elif name == "LinkedIn Message": # Simple string or None/Error from generate_linkedin_message
                if isinstance(result, str) and not result.startswith("Error generating message"):
                    st.session_state.generation_results[name] = {"text": result}
                    # No validation needed/queued for LinkedIn message
                else: # Likely None or error string from handler
                    error_msg = result if isinstance(result, str) else "LinkedIn message generation failed."
                    st.session_state.generation_results[name] = {"error": error_msg}

            else: # Resume or Cover Letter (Dict result from _collect_stream)
                if isinstance(result, dict):
                    if result.get("error"):
                        st.session_state.generation_results[name] = result # Store error dict
                    elif result.get("text"):
                        st.session_state.generation_results[name] = result # Store text dict
                        # Queue validation task using the collected text
                        logging.info(f"Queueing validation task for {name}...")
                        if api_key: # Ensure api key is available for validation call
                            # Ensure jd_data_for_validation exists before queueing validation
                            if jd_data_for_validation:
                                validation_tasks[name] = asyncio.create_task(
                                    llm_handler._validate_ats_friendliness(
                                        document_text=result["text"],
                                        document_type=name,
                                        job_description_data=jd_data_for_validation,
                                        google_api_key=api_key
                                    )
                                )
                            else:
                                st.session_state.validation_results[name] = {"error": "Cannot run validation: Job Description data was not successfully prepared."}
                                logging.warning(f"Skipping validation for {name}: JD data missing from prep_data.")
                        else:
                            st.session_state.validation_results[name] = {"error": "API Key missing, cannot run validation."}
                    else:
                        st.session_state.generation_results[name] = {"error": "Stream completed but no text found."}
                else:
                    st.session_state.generation_results[name] = {"error": f"Stream task bad format: {type(result)}"}

        # --- Run Validation Tasks (if any were queued) ---
        if validation_tasks:
            status_placeholder.info(f"🔍 Running {len(validation_tasks)} ATS Validations...")
            try:
                validation_task_results = await asyncio.gather(*validation_tasks.values(), return_exceptions=True)
                validation_all_results = dict(zip(validation_tasks.keys(), validation_task_results))

                for name, val_result in validation_all_results.items():
                    if isinstance(val_result, Exception):
                        st.session_state.validation_results[name] = {"error": f"Validation task failed: {val_result}"}
                        logging.error(f"Validation task for {name} failed: {val_result}", exc_info=val_result)
                    elif isinstance(val_result, dict):
                        st.session_state.validation_results[name] = val_result # Store the validation dict
                    else:
                        st.session_state.validation_results[name] = {"error": f"Validation task returned unexpected type: {type(val_result)}"}
            except Exception as e_val_gather:
                logging.error(f"Error gathering validation tasks: {e_val_gather}", exc_info=True)
                # Assign error to any validation results not already populated
                for name in validation_tasks.keys():
                    if name not in st.session_state.validation_results:
                        st.session_state.validation_results[name] = {"error": f"Failed gathering validation: {e_val_gather}"}

        status_placeholder.empty() # Clear final status message


    # --- Application UI ---
    st.title("🏆 Hyper-Optimized ATS Content Suite")
    st.markdown(f"Generates tailored documents & LinkedIn messages designed to pass ATS and impress recruiters. Uses `{os.path.basename(DEFAULT_RESUME_TXT_PATH)}` as resume fallback.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        google_api_key = get_api_key()
        st.markdown("---")
        st.subheader("📄 Select Content:") # Renamed section slightly
        gen_resume = st.checkbox("Tailored Resume", value=True, key="gen_resume_cb")
        gen_cover_letter = st.checkbox("Cover Letter", value=True, key="gen_cl_cb")
        gen_email = st.checkbox("Email", value=True, key="gen_email_cb")
        gen_linkedin = st.checkbox("LinkedIn Referral Msg", value=True, key="gen_linkedin_cb") # Added LinkedIn
        st.markdown("---")
        st.subheader("✍️ Generation Options:")
        tone = st.selectbox( "Select Tone:", ("Professional", "Enthusiastic", "Formal", "Confident", "Data-driven"), key="tone_select" )
        email_recipient_type = st.selectbox( "Email Recipient Type:", (llm_handler.RECIPIENT_TA_HM, llm_handler.RECIPIENT_GENERAL), key="email_recipient_select", help="Select primary email audience." )
        linkedin_connection_name = st.text_input("LinkedIn Connection's Name (Optional):", key="linkedin_name_input", placeholder="e.g., John Smith", help="For personalizing LinkedIn message.")
        st.markdown("---")
        st.subheader("💾 Resume Input Source")
        st.caption(f"Priority: Upload ➔ Paste ➔ Default (`{os.path.basename(DEFAULT_RESUME_TXT_PATH)}`)")
        # Placeholder for resume source feedback - Create it here so it always exists
        if 'resume_source_feedback' not in st.session_state:
            st.session_state.resume_source_feedback = st.empty()
        st.markdown("---")
        st.info("Provide accurate JD and Resume inputs.")


    # --- Main Area Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Candidate Information")
        candidate_name = st.text_input("Candidate Name:", key="candidate_name_input", placeholder="e.g., Jane Doe")
        candidate_email = st.text_input("Candidate Email:", key="candidate_email_input", placeholder="e.g., jane.doe@email.com")
        st.subheader("📄 Resume Input")
        resume_text_area = st.text_area( f"Paste Resume Text Here:", height=300, placeholder="Paste resume...", key="resume_text_area_input", help="Used if no file uploaded." )
        uploaded_resume = st.file_uploader( "Or Upload Resume File (.txt):", type=["txt"], key="resume_file_uploader", help=f"Overrides pasted text & default." )
    with col2:
        st.subheader("🎯 Target Job Description")
        job_desc = st.text_area( "Paste Job Description Here:", height=355, # Reduced height slightly
            placeholder="Paste the full job description text...", key="job_desc_input"
        )
        # --- Job Link Input ---
        job_link = st.text_input("Job Posting Link (Optional):", key="job_link_input", placeholder="https://www.linkedin.com/jobs/view/...", help="Include link in generated email if provided.")
        # --- End Job Link Input ---


    # --- Generate Button ---
    st.markdown("---")
    generate_button = st.button("✨ Generate Selected Content", type="primary", use_container_width=True) # Updated button text

    # --- State Initialization ---
    # Initialize state variables if they don't exist
    if 'generation_results' not in st.session_state: st.session_state.generation_results = {}
    if 'validation_results' not in st.session_state: st.session_state.validation_results = {}
    if 'prep_data' not in st.session_state: st.session_state.prep_data = {}
    if 'run_key' not in st.session_state: st.session_state.run_key = 0
    if 'missing_keywords_identified' not in st.session_state: st.session_state.missing_keywords_identified = []


    # --- Button Click Handler ---
    if generate_button:
        st.session_state.run_key += 1
        # Clear previous run's results and state
        st.session_state.generation_results = {}
        st.session_state.validation_results = {}
        st.session_state.prep_data = {}
        st.session_state.missing_keywords_identified = []
        # Ensure the feedback placeholder widget exists before trying to clear it
        if hasattr(st.session_state, 'resume_source_feedback'):
            st.session_state.resume_source_feedback.empty() # Clear previous source feedback

        # --- Determine Resume Content (Sync) ---
        final_resume_content = None; resume_source = "None Available"
        try: # Determine resume source
            if uploaded_resume is not None:
                stringio = StringIO(uploaded_resume.getvalue().decode("utf-8"))
                final_resume_content = stringio.read().strip()
                if final_resume_content: resume_source = "Uploaded File"
                else: st.warning("Uploaded resume file is empty."); resume_source = "Uploaded (Empty)" # Provide feedback but content is None
            if final_resume_content is None and resume_text_area.strip():
                final_resume_content = resume_text_area.strip(); resume_source = "Pasted Text"
            if final_resume_content is None:
                # Read from default file path
                final_resume_content = read_text_file(DEFAULT_RESUME_TXT_PATH)
                if final_resume_content: resume_source = f"Default File ({os.path.basename(DEFAULT_RESUME_TXT_PATH)})"
                # else: resume_source remains "None Available" (error handled in read_text_file)
        except Exception as e:
            st.error(f"Error processing resume input: {e}")
            logging.error(f"Error processing resume input: {e}", exc_info=True)
            final_resume_content = None; resume_source = "Error Reading Input"

        # Display resume source feedback using the placeholder created in the sidebar setup
        if hasattr(st.session_state, 'resume_source_feedback'):
            if resume_source not in ["None Available", "Error Reading Input", "Uploaded (Empty)"]:
                st.session_state.resume_source_feedback.success(f"Using resume from: **{resume_source}**", icon="💾")
            elif resume_source != "Unknown": # Show error/warning if determination finished with issue
                st.session_state.resume_source_feedback.error(f"Resume source issue: {resume_source}", icon="❌")


        # --- Input Validation (Sync) ---
        validation_passed = True
        selected_count = sum([gen_resume, gen_cover_letter, gen_email, gen_linkedin]) # Added LinkedIn
        if not google_api_key: st.error("API Key missing.", icon="❌"); validation_passed = False
        if selected_count == 0: st.error("Select at least one content type.", icon="⚠️"); validation_passed = False
        if not job_desc.strip(): st.error("Job Description required.", icon="❌"); validation_passed = False
        # Check resume content only if a document requiring it is selected
        if (gen_resume or gen_cover_letter or gen_email or gen_linkedin) and not final_resume_content:
            st.error(f"Resume Content required but none found/read.", icon="❌"); validation_passed = False
        if gen_email and not email_recipient_type: st.error("Email Recipient Type required.", icon="❌"); validation_passed = False
        # Relax name/email checks to warnings only
        if not candidate_name.strip(): st.warning("Candidate Name recommended.", icon="💡")
        if not candidate_email.strip(): st.warning("Candidate Email recommended.", icon="💡")
        # -- End Validation --

        # --- Define and Execute Main Async Orchestration ---
        async def async_main():
            prep_placeholder = st.empty()
            prep_data = {} # Initialize prep_data for this run
            is_fatal_error = False # Flag for prep errors

            try:
                # --- Configure API Key ---
                if google_api_key:
                    try:
                        genai.configure(api_key=google_api_key)
                        logging.info("Google GenAI configured within async_main.")
                    except Exception as config_e:
                        st.error(f"Failed to configure Google GenAI: {config_e}")
                        logging.error(f"Failed to configure Google GenAI in async_main: {config_e}", exc_info=True)
                        prep_placeholder.empty()
                        return # Stop if configuration fails
                else:
                    st.error("API Key missing, cannot configure Google GenAI.")
                    prep_placeholder.empty()
                    return

                # --- Run Data Prep Only If Needed (and only ONCE) ---
                # Data prep is needed if any document generation is selected
                needs_prep = gen_resume or gen_cover_letter or gen_email or gen_linkedin
                if needs_prep:
                    prep_placeholder.info("⚙️ Preparing & analyzing inputs...")
                    # Call data prep ONCE here
                    prep_data = await llm_handler._prepare_common_data(job_desc, final_resume_content, google_api_key)
                    st.session_state.prep_data = prep_data # Store result in session state
                    prep_placeholder.empty()

                    # Check for critical errors from prep before proceeding
                    prep_error = prep_data.get("error")
                    if prep_error:
                        # Determine if error is fatal or just warning
                        errors = prep_error.split('; ')
                        fatal_keywords = ["failed", "unavailable", "missing", "exception", "critical"]
                        # Treat as fatal if any error contains fatal keywords AND is not explicitly a warning
                        if any(fk in e.lower() for e in errors for fk in fatal_keywords if "warning" not in e.lower()):
                            is_fatal_error = True

                        if is_fatal_error:
                            st.error(f"Data Preparation Failed: {prep_error}. Cannot generate documents.")
                            logging.error(f"Data Preparation Failed: {prep_error}")
                        else: # Treat as warnings
                            st.warning(f"Data Preparation Warning: {prep_error}. Attempting generation...")

                    # Store and Display Missing Keywords (always do this after prep attempt)
                    st.session_state.missing_keywords_identified = prep_data.get("missing_keywords_from_resume", [])
                    if not prep_error: # Only show success/info message if no errors/warnings
                        if st.session_state.missing_keywords_identified:
                            st.info(f"JD keywords potentially missing: `{', '.join(st.session_state.missing_keywords_identified)}` (Attempting incorporation).")
                        else:
                            st.success("✅ Input analysis complete. No specific missing resume keywords targeted.", icon="🔍")
                        await asyncio.sleep(0.5) # Short delay for user to see message
                    elif st.session_state.missing_keywords_identified and is_fatal_error: # Show even if fatal error
                         st.warning(f"Data prep failed, but identified potential missing keywords: `{', '.join(st.session_state.missing_keywords_identified)}`")


                else: # No prep needed
                    prep_placeholder.empty()

                # --- Create Async Tasks ---
                tasks = {}
                # Only create tasks if prep was not needed, or if it was needed and didn't have a fatal error
                allow_task_creation = not needs_prep or (needs_prep and not is_fatal_error)

                if allow_task_creation:
                    # *** CORRECTED Task Creation Calls ***
                    if gen_resume:
                        tasks["Resume"] = asyncio.create_task(_collect_stream(
                            llm_handler.generate_application_text_streamed( # Pass raw inputs
                                name=candidate_name or "Candidate",
                                email=candidate_email or "candidate@example.com",
                                job_description=job_desc,                 # Pass raw JD
                                resume_content=final_resume_content,      # Pass raw Resume
                                generation_type=llm_handler.TYPE_RESUME,
                                google_api_key=google_api_key,
                                tone=tone
                            )
                        ))
                    if gen_cover_letter:
                        tasks["Cover Letter"] = asyncio.create_task(_collect_stream(
                            llm_handler.generate_application_text_streamed( # Pass raw inputs
                                name=candidate_name or "Candidate",
                                email=candidate_email or "candidate@example.com",
                                job_description=job_desc,                 # Pass raw JD
                                resume_content=final_resume_content,      # Pass raw Resume
                                generation_type=llm_handler.TYPE_COVER_LETTER,
                                google_api_key=google_api_key,
                                tone=tone
                            )
                        ))
                    if gen_email:
                         tasks["Email"] = asyncio.create_task(
                            llm_handler.generate_email_and_validate( # Pass raw inputs
                                name=candidate_name or "Candidate",
                                email=candidate_email or "candidate@example.com",
                                job_description=job_desc,                 # Pass raw JD
                                resume_content=final_resume_content,      # Pass raw Resume
                                google_api_key=google_api_key,
                                tone=tone,
                                email_recipient_type=email_recipient_type,
                                job_link=job_link or None
                            )
                        )
                    if gen_linkedin:
                        # generate_linkedin_message needs the *processed* dicts from prep_data
                        jd_data_for_linkedin = prep_data.get("jd_data", {})
                        # Ensure we get the resume_data dict *within* prep_data
                        resume_data_for_linkedin = prep_data.get("resume_data", {})
                        # Also need ranked reqs from prep_data for LinkedIn
                        ranked_reqs_for_linkedin = prep_data.get("ranked_jd_requirements", [])
                        # Add ranked_reqs to the resume_data dict if not present (function expects it there)
                        if "ranked_jd_requirements" not in resume_data_for_linkedin:
                             resume_data_for_linkedin["ranked_jd_requirements"] = ranked_reqs_for_linkedin

                        tasks["LinkedIn Message"] = asyncio.create_task(
                             llm_handler.generate_linkedin_message(
                                 name=candidate_name or "Candidate",
                                 job_description_data=jd_data_for_linkedin, # Pass processed JD Dict
                                 resume_data=resume_data_for_linkedin,       # Pass processed Resume Dict (incl. ranked reqs)
                                 google_api_key=google_api_key,
                                 tone=tone,
                                 connection_name=linkedin_connection_name or None
                             )
                         )
                elif needs_prep and is_fatal_error:
                     # Prep needed but failed fatally, error already shown
                     pass # Don't create tasks
                elif not needs_prep: # No tasks selected
                     st.info("No content selected for generation.")

                # --- Run Tasks Concurrently ---
                if tasks:
                    # Pass prep_data to run_generation_tasks for validation step context
                    await run_generation_tasks(tasks, prep_data, google_api_key)
                # else: Conditions where no tasks run are handled above

            except Exception as main_e:
                st.error(f"An unexpected error occurred during asynchronous execution: {main_e}")
                logging.exception("Error in async_main execution:")
            finally:
                prep_placeholder.empty() # Ensure cleanup


        # --- Execute ---
        if validation_passed:
            st.info("Initiating generation...")
            try:
                asyncio.run(async_main()) # Run the orchestrator
                # Check if any task actually produced an error stored in session state
                has_errors = any(res.get("error") for res in st.session_state.generation_results.values()) or \
                             any(val.get("error") for val in st.session_state.validation_results.values())

                if not has_errors:
                    st.success("✅ Generation process finished!", icon="🎉")
                else:
                    st.warning("Generation process finished, but some errors occurred (see details below).", icon="⚠️")

            except Exception as e:
                st.error(f"Error running the main async process: {e}")
                logging.exception("Error calling asyncio.run(async_main):")
        else:
            st.error("Please fix validation errors before generating.", icon="🚫")


    # --- Display Outputs ---
    # Use a key derived from run_key to force redraw of this container block
    output_display_key = f"output_display_{st.session_state.run_key}"

    with st.container(key=output_display_key):
        # Only display if results exist for the current run_key trigger
        if st.session_state.generation_results and st.session_state.run_key > 0:
            st.markdown("---")
            st.subheader("✨ Generated Content & Analysis")
            output_order = ["Resume", "Cover Letter", "Email", "LinkedIn Message"] # Define display order
            # Get keywords identified as missing during prep for THIS run
            originally_missing_keywords = set(st.session_state.get("missing_keywords_identified", []))

            for name in output_order:
                # Check if generation was requested and results are available
                if name in st.session_state.generation_results:
                    result_data = st.session_state.generation_results[name]
                    validation_data = st.session_state.validation_results.get(name)
                    # Determine icon
                    icon_map = {"Resume": "📄", "Cover Letter": "✉️", "Email": "📧", "LinkedIn Message": "🔗"}
                    icon = icon_map.get(name, "✨")

                    with st.expander(f"{icon} **{name}**", expanded=True):
                        if result_data.get("error"):
                            st.error(f"Generation Error: {result_data['error']}")
                            # Show raw validation response only if validation also had an error and has raw data
                            if validation_data and validation_data.get("error") and validation_data.get("raw_response"):
                                with st.expander("Show Raw Validation Response (Debug)"):
                                    st.code(validation_data["raw_response"], language=None)

                        elif result_data.get("text"):
                            full_text = result_data["text"]

                            # --- Display LLM Notes ---
                            llm_notes = re.findall(r'\[Note: (.*?)\]', full_text)
                            if llm_notes:
                                st.warning(f"**LLM Notes/Assumptions:**", icon="⚠️")
                                for note in llm_notes: st.caption(f"- {note}")
                            edited_text_value = full_text # Keep notes visible in editor

                            # --- Inline Editing ---
                            text_area_height = 150 if name == "LinkedIn Message" else 400
                            edited_text = st.text_area(
                                f"Editable {name}:", value=edited_text_value, height=text_area_height,
                                key=f"{name}_output_area_{st.session_state.run_key}" # Unique key per run
                            )

                            # --- Character Count for LinkedIn ---
                            if name == "LinkedIn Message":
                                char_count = len(edited_text)
                                char_limit = 300
                                if char_count > char_limit: st.error(f"Character Count: {char_count}/{char_limit} (LinkedIn Limit Exceeded!)")
                                else: st.caption(f"Character Count: {char_count}/{char_limit}")
                            # --- End Character Count ---

                            # --- Download Button (Exclude LinkedIn) ---
                            if name != "LinkedIn Message":
                                download_filename = f"{candidate_name.replace(' ','_') or 'Candidate'}_{name.replace(' ','_')}_{time.strftime('%Y%m%d')}.txt"
                                try: download_data = edited_text.encode('utf-8') # Use potentially edited text
                                except Exception as enc_e: st.error(f"Encoding Error: {enc_e}"); download_data = full_text.encode('utf-8')
                                st.download_button(
                                    label=f"Download {name} (.txt)", data=download_data,
                                    file_name=download_filename, mime="text/plain",
                                    key=f"{name}_download_{st.session_state.run_key}" # Unique key per run
                                )

                            # --- Readability & Validation (Exclude LinkedIn) ---
                            if name != "LinkedIn Message":
                                st.markdown("---");
                                # Readability Score
                                try:
                                    readability_score = textstat.flesch_kincaid_grade(full_text) # Use original generated text
                                    st.metric(f"Readability (FK Grade):", f"{readability_score:.1f}", help="Lower grade level generally indicates easier readability.")
                                except Exception as read_e:
                                    logging.warning(f"Readability calc failed for {name}: {read_e}")
                                    st.caption("Readability score unavailable.")

                                # Display Validation Results
                                if validation_data:
                                    st.markdown("---")
                                    st.write(f"**📊 ATS Validation ({name})**")
                                    if validation_data.get("error"):
                                        st.error(f"Validation Error: {validation_data['error']}")
                                        if "raw_response" in validation_data:
                                            with st.expander("Show Raw Validation Response (Debug)"): st.code(validation_data["raw_response"], language=None)
                                    else:
                                        score = validation_data.get("ats_score", "N/A")
                                        # Score display & Progress Bar
                                        progress_value = 0.0
                                        if isinstance(score, (int, float)):
                                            try: score_float = float(score); progress_value = min(score_float, 5.0) / 5.0
                                            except: pass
                                        st.progress(progress_value)
                                        st.metric("ATS Score (1-5):", score)

                                        # Validation Details Columns
                                        col_val1, col_val2 = st.columns(2)
                                        with col_val1: # Keyword Info
                                            kw_check = validation_data.get("keyword_check", {})
                                            st.write("**Keyword Analysis:**")
                                            if isinstance(kw_check, dict):
                                                found_in_output = set(kw_check.get('found_keywords', []))
                                                incorporated = sorted(list(originally_missing_keywords.intersection(found_in_output)))
                                                st.caption(f"Density: *{kw_check.get('density_impression', 'N/A')}*")
                                                st.write("*JD Keywords Found:*"); st.caption(f"`{', '.join(sorted(list(found_in_output))) or 'None'}`")
                                                if originally_missing_keywords: # Only show if some were missing initially
                                                    st.write("*Successfully Incorporated Missing Keywords:*");
                                                    if incorporated: st.success(f"`{', '.join(incorporated)}`", icon="✅")
                                                    else: st.caption("None.")
                                            else: st.caption(f"{kw_check}")

                                        with col_val2: # Other Checks
                                            st.write(f"**Clarity/Structure:** {validation_data.get('clarity_structure_check', 'N/A')}")
                                            st.write(f"**Formatting:** {validation_data.get('formatting_check', 'N/A')}")
                                            detailed = validation_data.get("detailed_checks", {})
                                            if isinstance(detailed, dict) and detailed:
                                                st.write("**Detailed Checks:**");
                                                for check_name, check_result in detailed.items(): st.caption(f"- {check_name.replace('_',' ').title()}: {check_result}")

                                        st.markdown("**Overall Feedback:**"); st.info(validation_data.get('overall_feedback', 'N/A'))
                                # Handle case where validation should have run but is missing
                                elif not result_data.get("error"): st.warning("Validation results unavailable.")
                            # --- END VALIDATION DISPLAY ---
                        else: st.warning("No text content generated for this document.")


    # --- Footer ---
    st.markdown("---")
    try:
        current_date = time.strftime('%Y-%m-%d %H:%M:%S %Z')
        current_location = "Syracuse, NY, USA" # Updated March 26, 2025 context
        st.caption(f"Powered by Google Gemini | Review & Personalize | {current_date} ({current_location})")
    except Exception as e:
        st.caption(f"Powered by Google Gemini | Review & Personalize")
        logging.error(f"Error generating footer: {e}")

# --- Call the main function only when script is run directly ---
if __name__ == "__main__":
    main()