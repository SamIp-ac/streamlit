import streamlit as st
import pandas as pd

st.markdown("# Transform mxl file to speech")
st.sidebar.markdown("# mxl to wav speech convertor")


mxlfile = st.file_uploader("Upload a mxl file")
