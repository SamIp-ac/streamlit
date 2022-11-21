import streamlit as st
import pandas as pd

st.markdown("# Add widgets to sidebar")
st.sidebar.markdown("# Add widgets to sidebar")

# Just add it after st.sidebar:
a = st.sidebar.radio('Select one:', [1, 2])

# Or use "with" notation
with st.sidebar:
    st.radio('Select one:', [1, 2])