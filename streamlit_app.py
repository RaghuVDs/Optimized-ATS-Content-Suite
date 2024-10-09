import streamlit as st
from hw01 import hw01 
from hw02 import lab2 
from hw03 import lab3
from hw05 import hw05


with st.sidebar:
    selected_page = st.radio("Select a page", ["HW01", "HW02","HW03","HW05"])

# Display the selected page
if selected_page == "HW01":
    hw01()
elif selected_page == "HW02":
    lab2()
elif selected_page == "HW03":
    lab3()
elif selected_page == "HW05":
    hw05()