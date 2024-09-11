import streamlit as st
from openai import OpenAI
from openai import AuthenticationError
import PyPDF2
import requests
from bs4 import BeautifulSoup

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
            "<p style='text-align: center;'>Upload a document and ask questions about it – GPT will answer!</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center;'>To use this app, you need to provide an OpenAI API key, which you can get <a href='https://platform.openai.com/account/api-keys'>here</a>.</p>",
            unsafe_allow_html=True,
        )

        # API key handling
        openai_api_key = st.secrets["api_key"]
        if not openai_api_key:
            st.error("OpenAI API key not found in secrets.")
            st.stop()

        try:
            client = OpenAI(api_key=openai_api_key)

            # Sidebar with summary options and model choice
            with st.sidebar:
                st.subheader("Summary Options")
                summary_option = st.radio(
                    "Choose summary type:",
                    ("100 words", "2 paragraphs", "5 bullet points")
                )
                use_advanced_model = st.checkbox("Use Advanced Model (gpt-4)")
                model_name = "gpt-4" if use_advanced_model else "gpt-4-0314"

            # URL input at the top
            url = st.text_input("Enter a URL or upload a document below:")

            # File uploader
            uploaded_file = st.file_uploader(
                "Upload a Document (.txt, .md, or .pdf)", type=("txt", "md", "pdf")
            )

            if st.button("Generate Summary") and (url or uploaded_file):
                # Process the input (URL or file)
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

                # Construct prompt based on selected summary option
                if summary_option == "100 words":
                    prompt = f"Summarize the following document in 100 words:\n\n{document}"
                elif summary_option == "2 paragraphs":
                    prompt = f"Summarize the following document in 2 connecting paragraphs:\n\n{document}"
                else:  # 5 bullet points
                    prompt = f"Summarize the following document in 5 bullet points:\n\n{document}"

                # Generate summary
                try:
                    with st.spinner("Generating summary..."):
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )

                    # Display the summary only if it's generated
                    if response.choices[0].message.content:
                        st.subheader("Summary")
                        st.write(response.choices[0].message.content)
                except Exception as e:
                    st.exception(e) 

        except AuthenticationError:
            st.error("Invalid OpenAI API key. Please check your key and try again.")
