import streamlit as st
from openai import OpenAI
from openai import OpenAIError

import PyPDF2


def read_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def hw01():
        st.set_page_config(layout="wide")

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

        # STEP ONE: CHECKING API KEY VALIDITY FIRST
        openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here")

        if openai_api_key:  # AttemptS validation if a key is entered
            try:
                client = OpenAI(api_key=openai_api_key)
                models = client.models.list()  
                st.success("API key is valid!")
            except OpenAIError as e:
                st.error(f"Error validating API key: {e}")
                st.stop()  # Stop if the API key is invalid

        else:  # When no apikey enterdd
            st.warning("Please enter your OpenAI API key to proceed.")
            st.stop()

        # STEP TWO: PROCEESS WITH FILE UPLOAD IF API KEY IS VALID
        col1, col2 = st.columns([1, 2])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload a Document (.pdf or .txt)", 
                type=("pdf", "txt"),
                accept_multiple_files=False, 
                help="Supported formats: .pdf, .txt"
            )

            # file type and process file
            if uploaded_file is not None:
                if uploaded_file.type not in ("text/plain", "application/pdf"):
                    st.session_state['file_upload_warning'] = "Unsupported file type. Please upload a .pdf or .txt file." 
                    uploaded_file = None

                elif st.button(("Process File" if uploaded_file is not None else "No File Uploaded"), disabled=not uploaded_file):
                    with st.spinner("Processing file..."):
                        try:
                            if uploaded_file.type == "text/plain":
                                st.session_state['document'] = uploaded_file.read().decode() 
                            elif uploaded_file.type == "application/pdf":
                                st.session_state['document'] = read_pdf(uploaded_file) 

                            print("Document after processing:", st.session_state['document'])  
                            st.success("File processed successfully!")
                        except UnicodeDecodeError as e:
                            st.error(f"Error decoding text file: {e}")
                        except RuntimeError as e:
                            st.error(f"Error processing PDF file: {e}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}", icon="🚨")

            # warning from session state if exist
            if 'file_upload_warning' in st.session_state:
                st.warning(st.session_state['file_upload_warning'])
                del st.session_state['file_upload_warning']

        with col2:
            question = st.text_area(
                "Ask a Question",
                placeholder="Can you give me a short summary?",
                height=120,
                disabled=not uploaded_file 
            )

            if st.button("Get Answer", disabled=not ('document' in st.session_state and question and "Process File")): 
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": f"Here's a document: {st.session_state['document']} \n\n---\n\n {question}",
                        }
                    ]

                    with st.spinner("Generating answer..."):
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",  
                            messages=messages
                        )

                    st.write(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"An error occurred while processing the request: {e}")

                # Delete Button for removing docs from the meory
                if 'document' in st.session_state:
                    if st.button("Delete Document from Memory"):  
                        del st.session_state['document']
                        st.success("Document successfully deleted from memory.")