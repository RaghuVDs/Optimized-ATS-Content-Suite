import streamlit as st
from hw01 import hw01 
from hw02 import hw02 

st.set_page_config(page_title="Homework Manager", page_icon=":book:", layout="wide")


st.title("Homework Manager")

pg = st.navigation([
    st.Page(hw01, title="HW 1"),
    st.Page(hw02, title="HW 2")
], position="sidebar")

pg.run()