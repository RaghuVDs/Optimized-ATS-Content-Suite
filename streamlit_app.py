import streamlit as st
from hw01 import hw01 
from hw02 import lab2 

with st.sidebar:
    selected_page = st.radio("Select a page", ["HW01", "HW02"])

# Display the selected page
if selected_page == "HW01":
    hw01()
elif selected_page == "HW02":
    lab2()