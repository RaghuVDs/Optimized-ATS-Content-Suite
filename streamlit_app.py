import streamlit as st
from app import job 



with st.sidebar:
    selected_page = st.radio("Select a page", ["app"])

# Display the selected page
if selected_page == "app":
    job()