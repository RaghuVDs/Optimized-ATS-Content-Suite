import streamlit as st
from openai import OpenAI
from openai import AuthenticationError
import PyPDF2
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
from anthropic import Anthropic

def read_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

def lab2():
    st.markdown(
        "<h1 style='text-align: center;'>📄 Document Question Answering</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Upload a document or enter a URL to get a summary – GPT will help!</p>",
        unsafe_allow_html=True,
    )

    # API key handling for OpenAI
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    if not openai_api_key:
        st.error("OpenAI API key not found in secrets.")
        st.stop()

    # API key handling for Google Gemini 
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    if not google_api_key:
        st.error("Google API key not found in secrets.")
        st.stop()

    # API key handling for Anthropic
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
    if not anthropic_api_key:
        st.error("Anthropic API key not found in secrets.")
        st.stop()

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Initialize Anthropic client
        anthropic_client = Anthropic(api_key=anthropic_api_key)

        # Sidebar
        with st.sidebar:
            st.subheader("Summary Options")
            summary_option = st.radio(
                "Choose summary type:",
                ("100 words", "2 paragraphs", "5 bullet points")
            )

            st.subheader("Model")
            # Dropdown for LLM provider
            llm_provider = st.selectbox(
                "Select LLM Provider:",
                ("OpenAI", "Google Gemini", "Anthropic")
            )

            st.subheader("Language")
            output_language = st.selectbox(
                "Select output language:",
                ("English", "French", "Spanish", "Hindi", "Kannada")
            )

        # Map LLM options
        llm_mapping = {
            "gpt-4o": client,
            "gpt-4o-mini": client
        }

        # Initialize session state variables if they don't exist
        if 'input_method' not in st.session_state:
            st.session_state['input_method'] = None

        # URL input
        if st.session_state['input_method'] != 'file':
            url = st.text_input("Enter a URL or upload a document below:")
            if url:
                st.session_state['input_method'] = 'url'
        else:
            url = None

        # File uploader
        if st.session_state['input_method'] != 'url':
            uploaded_file = st.file_uploader(
                "Upload a Document (.txt, .md, or .pdf)", type=("txt", "md", "pdf")
            )
            if uploaded_file:
                st.session_state['input_method'] = 'file'
        else:
            uploaded_file = None

        if st.button("Generate Summary") and (url or uploaded_file):
            try:
                if url:
                    document = read_url_content(url)
                    if document is None:
                        st.error("Error fetching content from the URL. Please check the URL and try again.")
                        return
                elif uploaded_file:
                    if uploaded_file.type == "application/pdf":
                        document = read_pdf(uploaded_file)
                    else: 
                        document = uploaded_file.read().decode("utf-8") 

            except UnicodeDecodeError:
                try:
                    document = uploaded_file.read().decode("latin-1")
                except UnicodeDecodeError:
                    st.error("Error decoding file. Please ensure the file is in UTF-8 or Latin-1 encoding.")
                    return 
            except Exception as e:
                st.exception(e)
                return

            # summary option and language
            if summary_option == "100 words":
                prompt = f"Summarize the following document in 100 words in {output_language}:\n\n{document}"
            elif summary_option == "2 paragraphs":
                prompt = f"Summarize the following document in 2 connecting paragraphs in {output_language}:\n\n{document}"
            else:  # 5 bullet points
                prompt = f"Summarize the following document in 5 bullet points in {output_language}:\n\n{document}"

            # Generate summaries using both lower and higher-tier models for the selected provider
            try:
                with st.spinner("Generating summaries..."):
                    if llm_provider == "OpenAI":
                        lower_model = "gpt-4o-mini"
                        higher_model = "gpt-4o"

                        response_lower = client.chat.completions.create(
                            model=lower_model,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        summary_lower = response_lower.choices[0].message.content

                        response_higher = client.chat.completions.create(
                            model=higher_model,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        summary_higher = response_higher.choices[0].message.content

                    elif llm_provider == "Google Gemini":
                        lower_model = "gemini-1.5-flash" 
                        higher_model = "gemini-1.5-pro"

                        # Configure Google Generative AI
                        genai.configure(api_key=google_api_key)

                        google_model_lower = genai.GenerativeModel(model_name=lower_model)
                        response_lower = google_model_lower.generate_content(prompt)
                        summary_lower = response_lower.text

                        google_model_higher = genai.GenerativeModel(model_name=higher_model)
                        response_higher = google_model_higher.generate_content(prompt)
                        summary_higher = response_higher.text

                    elif llm_provider == "Anthropic":
                        lower_model = "Claude 3 Haiku" 
                        higher_model = "Claude 3.5 Sonnet"

                        response_lower = anthropic_client.completions.create(
                            model=lower_model,
                            prompt=prompt,
                            max_tokens_to_sample=1024, 
                        )
                        summary_lower = response_lower.completion

                        response_higher = anthropic_client.completions.create(
                            model=higher_model,
                            prompt=prompt,
                            max_tokens_to_sample=1024, 
                        )
                        summary_higher = response_higher

                        # Display the summaries
                if summary_lower and summary_higher:
                    st.subheader(f"Summary ({lower_model})")
                    st.write(summary_lower)

                    st.subheader(f"Summary ({higher_model})")
                    st.write(summary_higher)

            except Exception as e:
                st.exception(e)

    except AuthenticationError:
        st.error("Invalid OpenAI API key. Please check your key and try again.")