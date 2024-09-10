import streamlit as st
from hw01 import hw01 
from hw02 import hw02 

st.set_page_config(page_title="Homework Manager", page_icon=":book:", layout="wide")


st.title("Homework Manager")

with st.sidebar:
    selected_page = st.radio("Select a page", ["HW 1", "HW 2"])

if selected_page == "HW 1":
    hw01()
elif selected_page == "HW 2":
    hw02()