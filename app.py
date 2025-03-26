import streamlit as st
import llm_handler  # Import the updated module
import os
from io import StringIO
import time
import asyncio # Import asyncio
import re # For finding keywords later if needed, and cleaning text
import logging # Import logging
from typing import Optional, Dict, List, Any, AsyncGenerator # Only necessary typing imports

# --- Constants ---
DEFAULT_RESUME_TXT_PATH = "/workspaces/688-HW/default_resume.txt" # UPDATE THIS PATH if needed

# --- Page Configuration ---
st.set_page_config(
    page_title="ATS Content Suite Generator (No Highlight)",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [App] %(message)s')

# --- Helper Functions ---
def get_api_key():
    """Retrieves the Google API Key from secrets or user input."""
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        # st.sidebar.success("API Key loaded from Secrets.", icon="✅") # Reduce clutter
        return api_key
    except (FileNotFoundError, KeyError):
        st.sidebar.warning("API Key not found in Secrets.", icon="⚠️")
        api_key = st.sidebar.text_input("Enter Google API Key:", type="password", key="api_key_input")
        if not api_key:
            st.sidebar.error("API Key is required to run the generator.", icon="❌")
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
            return None
        return content.strip()
    except Exception as e:
        st.sidebar.error(f"Error reading default resume: {e}", icon="❌")
        logging.error(f"Error reading default resume file '{file_path}': {e}", exc_info=True)
        return None

# Removed get_jd_keywords and highlight_text functions

async def _collect_stream(async_generator: AsyncGenerator[str, None]) -> Dict[str, Any]:
    """Helper to collect text from an async stream generator and handle errors."""
    full_text_list = []
    error_found = None
    try:
        async for chunk in async_generator:
            if isinstance(chunk, str):
                if chunk.strip().startswith("--- ERROR"):
                    error_found = chunk.strip()
                    logging.error(f"Error message yielded by stream: {error_found}")
                    break
                full_text_list.append(chunk)
            else:
                logging.warning(f"Received non-string chunk from stream: {type(chunk)}")

        if error_found:
            collected_text = "".join(full_text_list)
            return {"error": f"{error_found}\n(Collected text before error: '{collected_text[:200]}...')"}

        full_text = "".join(full_text_list)
        full_text = re.sub(r'^--- Generating.*?---\n?', '', full_text, flags=re.MULTILINE)
        full_text = re.sub(r'\n?--- Generating.*?---\n?', '', full_text, flags=re.MULTILINE)

        if not full_text.strip():
             return {"error": "Generation stream completed but yielded no text content."}

        return {"text": full_text.strip()}

    except Exception as e:
        logging.error(f"Error collecting stream: {e}", exc_info=True)
        return {"error": f"Stream collection error: {e}"}

# --- Async Task Runner ---
async def run_generation_tasks(tasks_to_run: Dict[str, asyncio.Task], prep_data: Dict, api_key: str):
    """Runs asyncio tasks, stores results and validation in session state."""
    st.session_state.generation_results = {} # Reset results
    st.session_state.validation_results = {} # Reset validation

    spinners = {}
    overall_spinner_placeholder = st.empty()
    overall_spinner_placeholder.info("🚀 Starting asynchronous generation tasks...")
    status_placeholder = st.container()

    for name in tasks_to_run.keys():
         spinners[name] = status_placeholder.empty()
         spinners[name].info(f"⏳ Queued {name} generation...")

    # Gather results concurrently
    results = await asyncio.gather(*tasks_to_run.values(), return_exceptions=True)
    all_results = dict(zip(tasks_to_run.keys(), results))

    overall_spinner_placeholder.empty()

    # --- Process & Validate Results ---
    validation_tasks = {}
    jd_data_for_validation = prep_data.get("jd_data", {}) # Get JD data once

    for name, result in all_results.items():
        spinners[name].empty()
        status_placeholder.info(f"⚙️ Processing result for {name}...")

        if isinstance(result, Exception):
            error_msg = f"Task failed: {result}"
            st.session_state.generation_results[name] = {"error": error_msg}
            logging.error(f"Task {name} failed in gather: {result}", exc_info=result)

        elif name == "Email":
             # Email task returns a tuple: (email_text, validation_dict)
             if isinstance(result, tuple) and len(result) == 2:
                 email_text, validation_dict = result
                 if email_text:
                     st.session_state.generation_results[name] = {"text": email_text}
                     st.session_state.validation_results[name] = validation_dict or {"warning": "No validation data returned by email task."}
                 else:
                     error_msg = validation_dict.get("error", "Email generation returned None.") if validation_dict else "Email generation failed."
                     st.session_state.generation_results[name] = {"error": error_msg}
                     if validation_dict: st.session_state.validation_results[name] = validation_dict
             else:
                  st.session_state.generation_results[name] = {"error": "Email task returned unexpected result format."}

        else: # Resume or Cover Letter (Result is dict from _collect_stream)
             if isinstance(result, dict):
                 if result.get("error"):
                     st.session_state.generation_results[name] = result # Store the error dict
                 elif result.get("text"):
                      st.session_state.generation_results[name] = result # Store the text dict
                      # Queue validation task for this streamed content
                      logging.info(f"Queueing validation task for {name}...")
                      validation_tasks[name] = asyncio.create_task(
                          llm_handler._validate_ats_friendliness(
                              document_text=result["text"],
                              document_type=name,
                              job_description_data=jd_data_for_validation,
                              google_api_key=api_key
                          )
                      )
                 else:
                     st.session_state.generation_results[name] = {"error": "Stream completed but returned no text or error."}
             else:
                st.session_state.generation_results[name] = {"error": f"Task returned unexpected result type: {type(result)}"}

    # --- Run Validation Tasks for Streamed Content ---
    if validation_tasks:
        status_placeholder.info("🔍 Running ATS Validations for Resume/Cover Letter...")
        try:
            validation_task_results = await asyncio.gather(*validation_tasks.values(), return_exceptions=True)
            validation_all_results = dict(zip(validation_tasks.keys(), validation_task_results))

            for name, val_result in validation_all_results.items():
                if isinstance(val_result, Exception):
                     st.session_state.validation_results[name] = {"error": f"Validation task failed: {val_result}"}
                     logging.error(f"Validation task for {name} failed: {val_result}", exc_info=val_result)
                elif isinstance(val_result, dict):
                     st.session_state.validation_results[name] = val_result
                else:
                     st.session_state.validation_results[name] = {"error": f"Validation task returned unexpected type: {type(val_result)}"}
        except Exception as e_val_gather:
             logging.error(f"Error gathering validation tasks: {e_val_gather}", exc_info=True)
             for name in validation_tasks.keys():
                 if name not in st.session_state.validation_results:
                      st.session_state.validation_results[name] = {"error": f"Failed to gather validation results: {e_val_gather}"}

    status_placeholder.empty() # Clear status messages


# --- Application UI ---
st.title("🚀 Advanced ATS Content Suite Generator")
st.markdown(f"Generate tailored application documents asynchronously. Uses `{os.path.basename(DEFAULT_RESUME_TXT_PATH)}` as fallback.")

with st.sidebar:
    st.header("⚙️ Configuration")
    google_api_key = get_api_key()
    st.markdown("---")
    st.subheader("📄 Select Documents:")
    gen_resume = st.checkbox("Tailored Resume", value=True, key="gen_resume_cb")
    gen_cover_letter = st.checkbox("Cover Letter", value=True, key="gen_cl_cb")
    gen_email = st.checkbox("Email", value=True, key="gen_email_cb")
    st.markdown("---")
    st.subheader("✍️ Generation Options:")
    tone = st.selectbox(
        "Select Tone:",
        ("Professional", "Enthusiastic", "Formal", "Confident", "Data-driven"),
        key="tone_select"
    )
    email_recipient_type = st.selectbox(
        "Email Recipient Type:",
        (llm_handler.RECIPIENT_TA_HM, llm_handler.RECIPIENT_GENERAL),
        key="email_recipient_select",
        help="Select who the email is primarily intended for."
    )
    st.markdown("---")
    st.subheader("💾 Resume Input Source")
    st.caption(f"Priority: Uploaded `.txt` ➔ Pasted Text ➔ Default (`{os.path.basename(DEFAULT_RESUME_TXT_PATH)}`)")
    st.session_state.resume_source_feedback = st.empty() # Placeholder for source feedback
    st.markdown("---")
    st.info("Provide accurate Job Description and Resume inputs for best results.")


col1, col2 = st.columns(2)
with col1:
    st.subheader("👤 Candidate Information")
    candidate_name = st.text_input("Candidate Name:", key="candidate_name_input", placeholder="e.g., Jane Doe")
    candidate_email = st.text_input("Candidate Email:", key="candidate_email_input", placeholder="e.g., jane.doe@email.com")

    st.subheader("📄 Resume Input")
    resume_text_area = st.text_area(
        f"Paste Resume Text Here:",
        height=300, placeholder="Paste the full text content of your resume...", key="resume_text_area_input",
        help="This will be used if no file is uploaded."
    )
    uploaded_resume = st.file_uploader(
        "Or Upload Resume File (Plain Text `.txt`):",
        type=["txt"], key="resume_file_uploader",
        help=f"Overrides pasted text and the default file ({os.path.basename(DEFAULT_RESUME_TXT_PATH)})."
    )
with col2:
    st.subheader("🎯 Target Job Description")
    job_desc = st.text_area(
        "Paste Job Description Here:", height=440,
        placeholder="Paste the full job description text...", key="job_desc_input"
    )

# --- Generate Button ---
st.markdown("---")
generate_button = st.button("✨ Generate Selected Documents", type="primary", use_container_width=True)

# --- State Initialization ---
if 'generation_results' not in st.session_state: st.session_state.generation_results = {}
if 'validation_results' not in st.session_state: st.session_state.validation_results = {}
# Removed jd_keywords from state as it's not used for highlighting anymore
if 'prep_data' not in st.session_state: st.session_state.prep_data = {}
if 'run_key' not in st.session_state: st.session_state.run_key = 0

# --- Button Click Handler ---
if generate_button:
    st.session_state.run_key += 1
    st.session_state.generation_results = {} # Clear previous results
    st.session_state.validation_results = {}
    st.session_state.prep_data = {}
    if 'resume_source_feedback' in st.session_state: # Check if placeholder exists
        st.session_state.resume_source_feedback.empty() # Clear previous source feedback

    # --- Determine Resume Content (Sync) ---
    final_resume_content = None
    resume_source = "None Available"
    try:
        if uploaded_resume is not None:
            stringio = StringIO(uploaded_resume.getvalue().decode("utf-8"))
            final_resume_content = stringio.read().strip()
            if final_resume_content:
                resume_source = "Uploaded File"
            else:
                st.warning("Uploaded resume file is empty.")
                resume_source = "Uploaded (Empty)"
        if final_resume_content is None and resume_text_area.strip():
            final_resume_content = resume_text_area.strip()
            resume_source = "Pasted Text"
        if final_resume_content is None:
            # Read from default file path
            final_resume_content = read_text_file(DEFAULT_RESUME_TXT_PATH)
            if final_resume_content:
                resume_source = f"Default File ({os.path.basename(DEFAULT_RESUME_TXT_PATH)})"
            # else: resume_source remains "None Available" (error handled in read_text_file)

    except Exception as e:
        st.error(f"Error processing resume input: {e}")
        logging.error(f"Error processing resume input: {e}", exc_info=True)
        final_resume_content = None
        resume_source = "Error Reading Input"

    # Display resume source feedback (Check if feedback widget exists)
    if hasattr(st.session_state, 'resume_source_feedback'):
        if resume_source not in ["None Available", "Error Reading Input", "Uploaded (Empty)"]:
            st.session_state.resume_source_feedback.success(f"Using resume from: **{resume_source}**", icon="💾")
        elif resume_source == "None Available":
            st.session_state.resume_source_feedback.error("No resume source available!", icon="❌")
        elif resume_source == "Uploaded (Empty)":
             st.session_state.resume_source_feedback.warning("Using empty uploaded file.", icon="⚠️")


    # --- Input Validation (Sync) ---
    validation_passed = True
    selected_count = sum([gen_resume, gen_cover_letter, gen_email])
    if not google_api_key: st.error("API Key missing.", icon="❌"); validation_passed = False
    if selected_count == 0: st.error("Select at least one document.", icon="⚠️"); validation_passed = False
    if not job_desc.strip(): st.error("Job Description required.", icon="❌"); validation_passed = False
    if (gen_resume or gen_cover_letter or gen_email) and not final_resume_content:
         st.error(f"Resume Content required but none found/read.", icon="❌"); validation_passed = False
    if gen_email and not email_recipient_type: st.error("Email Recipient Type required.", icon="❌"); validation_passed = False
    if not candidate_name.strip(): st.warning("Candidate Name recommended.", icon="💡")
    if not candidate_email.strip(): st.warning("Candidate Email recommended.", icon="💡")
    # -- End Validation --

    # --- Define and Execute Main Async Orchestration ---
    async def async_main():
        prep_placeholder = st.empty()
        prep_data = {}
        try:
            prep_placeholder.info("⚙️ Preparing & analyzing inputs...")
            prep_data = await llm_handler._prepare_common_data(job_desc, final_resume_content, google_api_key)
            st.session_state.prep_data = prep_data
            prep_placeholder.empty()

            if prep_data.get("error"):
                 st.error(f"Data Preparation Failed: {prep_data['error']}. Cannot generate documents.")
                 logging.error(f"Data Preparation Failed: {prep_data['error']}")
                 return

            if prep_data.get("warning"):
                 st.warning(prep_data["warning"])

            st.success("✅ Inputs analyzed.", icon="🔍")

            tasks = {}
            common_args = {
                "name": candidate_name or "Candidate",
                "email": candidate_email or "candidate@example.com",
                "job_description": job_desc, "resume_content": final_resume_content,
                "google_api_key": google_api_key, "tone": tone
            }

            if gen_resume:
                tasks["Resume"] = asyncio.create_task(
                    _collect_stream(llm_handler.generate_application_text_streamed(
                        **common_args, generation_type=llm_handler.TYPE_RESUME
                    ))
                )
            if gen_cover_letter:
                tasks["Cover Letter"] = asyncio.create_task(
                     _collect_stream(llm_handler.generate_application_text_streamed(
                        **common_args, generation_type=llm_handler.TYPE_COVER_LETTER
                    ))
                )
            if gen_email:
                tasks["Email"] = asyncio.create_task(
                    llm_handler.generate_email_and_validate(
                        **common_args, email_recipient_type=email_recipient_type
                    )
                )

            if tasks:
                await run_generation_tasks(tasks, prep_data, google_api_key)
            # else case handled by validation

        except Exception as main_e:
             prep_placeholder.empty()
             st.error(f"An unexpected error occurred during asynchronous execution: {main_e}")
             logging.exception("Error in async_main execution:")

    if validation_passed:
        st.info(f"Initiating generation...")
        try:
            asyncio.run(async_main())
            st.success("✅ Generation process complete!", icon="🎉")
        except Exception as e:
             st.error(f"Error running the main async process: {e}")
             logging.exception("Error calling asyncio.run(async_main):")
    else:
        st.error("Please fix validation errors before generating.", icon="🚫")


# --- Display Outputs ---
output_display_key = f"output_display_{st.session_state.run_key}"

with st.container(key=output_display_key):
    if st.session_state.generation_results:
        st.markdown("---")
        st.subheader("✨ Generated Documents & Validation")
        output_order = ["Resume", "Cover Letter", "Email"]

        for name in output_order:
            if name in st.session_state.generation_results:
                result_data = st.session_state.generation_results[name]
                validation_data = st.session_state.validation_results.get(name)
                icon = "📄" if name == "Resume" else "✉️" if name == "Cover Letter" else "📧"

                with st.expander(f"{icon} **{name}**", expanded=True):
                    if result_data.get("error"):
                        st.error(f"Generation Error: {result_data['error']}")
                        # Show raw validation response only if validation also had an error and has raw data
                        if validation_data and validation_data.get("error") and validation_data.get("raw_response"):
                            with st.expander("Show Raw Validation Response (Debug)"):
                                st.code(validation_data["raw_response"], language=None)

                    elif result_data.get("text"):
                        full_text = result_data["text"]

                        # Editable Text Area (NO highlighting)
                        edited_text = st.text_area(
                            f"Editable {name}:",
                            value=full_text, # Display plain text
                            height=400,
                            key=f"{name}_output_area_{st.session_state.run_key}"
                        )

                        # Download Button
                        download_filename = f"{candidate_name.replace(' ','_') or 'Candidate'}_{name.replace(' ','_')}_{time.strftime('%Y%m%d')}.txt"
                        try:
                           download_data = edited_text.encode('utf-8')
                        except Exception as enc_e:
                           st.error(f"Error encoding text for download: {enc_e}")
                           download_data = full_text.encode('utf-8') # Fallback

                        st.download_button(
                            label=f"Download {name} (.txt)",
                            data=download_data,
                            file_name=download_filename,
                            mime="text/plain",
                            key=f"{name}_download_{st.session_state.run_key}"
                        )

                        # --- DISPLAY VALIDATION RESULTS ---
                        if validation_data:
                            st.markdown("---")
                            st.write(f"**📊 ATS Validation ({name})**")
                            if validation_data.get("error"):
                                st.error(f"Validation Error: {validation_data['error']}")
                                if "raw_response" in validation_data:
                                     with st.expander("Show Raw Validation Response (Debug)"):
                                         st.code(validation_data["raw_response"], language=None)
                            else:
                                score = validation_data.get("ats_score", "N/A")
                                # Display score and progress bar
                                progress_value = 0.0
                                if isinstance(score, (int, float)):
                                    try:
                                        score_float = float(score)
                                        progress_value = min(score_float, 5.0) / 5.0
                                    except: pass # Keep progress 0 if conversion fails
                                st.progress(progress_value)
                                st.metric("ATS Score (1-5):", score)

                                # --- Validation Details (Modified Keyword Section) ---
                                col_val1, col_val2 = st.columns(2)
                                with col_val1:
                                    kw_check = validation_data.get("keyword_check", {})
                                    st.write("**Keyword Analysis:**")
                                    if isinstance(kw_check, dict):
                                        # Display Keywords Included from JD
                                        included_keywords = kw_check.get('found_keywords', [])
                                        st.write("*JD Keywords Included:*") # Label
                                        if included_keywords:
                                            st.caption(f"`{', '.join(included_keywords)}`")
                                        else:
                                            st.caption("None specifically identified from JD.")

                                        # Optionally keep suggestions for missing keywords
                                        missing_keywords = kw_check.get('missing_suggestions', [])
                                        st.write("*JD Keywords Missing (Suggestions):*") # Label
                                        if missing_keywords:
                                             st.caption(f"`{', '.join(missing_keywords)}`")
                                        else:
                                             st.caption("No specific missing keywords suggested.")

                                        st.write("*Keyword Density Impression:*") # Label
                                        st.caption(f"*{kw_check.get('density_impression', 'N/A')}*")

                                    else:
                                        st.caption(f"Keyword Info: {kw_check}") # Fallback

                                with col_val2: # Clarity, Structure, Formatting
                                    st.write(f"**Clarity/Structure:**")
                                    st.caption(f"{validation_data.get('clarity_structure_check', 'N/A')}")
                                    st.write(f"**Formatting:**")
                                    st.caption(f"{validation_data.get('formatting_check', 'N/A')}")

                                st.markdown("**Overall Feedback:**") # Feedback
                                st.info(validation_data.get('overall_feedback', 'N/A'))
                                # --- End Validation Details ---
                        else:
                            if not result_data.get("error"):
                                 st.warning("Validation results are pending or unavailable.")
                        # --- END VALIDATION DISPLAY ---
                    else:
                         st.warning("No text content found in the generation result.")

# --- Footer ---
st.markdown("---")
try:
    # Syracuse, New York, United States is in EDT on March 26, 2025
    # Using system time for simplicity, assuming server/local time is acceptable
    current_date = time.strftime('%Y-%m-%d %H:%M:%S %Z') # Add Timezone if available
    current_location = "Syracuse, NY, USA"
    st.caption(f"Powered by Google Gemini | Review & Personalize All Content | {current_date} ({current_location})")
except Exception as e:
     st.caption(f"Powered by Google Gemini | Review & Personalize All Content")
     logging.error(f"Error generating footer: {e}")