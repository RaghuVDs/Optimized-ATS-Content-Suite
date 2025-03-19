# pdf_handler.py
import PyPDF2
import streamlit as st
from typing import Optional

def read_pdf(uploaded_file) -> Optional[str]:
    """Reads text content from a PDF file.

    Args:
        uploaded_file: An uploaded file object (from Streamlit's file_uploader).

    Returns:
        The extracted text content as a string, or None if an error occurs.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None