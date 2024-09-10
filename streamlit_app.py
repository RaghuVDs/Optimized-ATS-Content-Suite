import streamlit as st
from hw01 import hw01 
from hw02 import hw02  

st.title("HomeWork Manager")

pg = st.navigation([
    st.Page(hw01, title="HW 1"),
    st.Page(hw02, title="HW 2")
], position="sidebar")

pg.run()