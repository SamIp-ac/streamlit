import streamlit as st
import pandas as pd
import mxl2wav as mw
st.markdown("# Add widgets to sidebar")
st.sidebar.markdown("# Add widgets to sidebar")


mxlfile = st.file_uploader("Upload a mxl file")
if mxlfile:
    ans = mw.m2w(mxlfile)
    st.download_button(
        label="Download data as txt",
        data=ans,
        file_name='audio.wav'
    )
'''# Just add it after st.sidebar:
a = st.sidebar.radio('Select one:', [1, 2])

# Or use "with" notation
with st.sidebar:
    st.radio('Select one:', [1, 2])'''

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.horizontal = False

col1, col2 = st.columns(2)

with col1:
    st.checkbox("Disable radio widget", key="disabled")
    st.checkbox("Orient radio options horizontally", key="horizontal")

with col2:
    st.radio(
        "Set label visibility 👇",
        ["visible", "hidden", "collapsed"],
        key="visibility",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        horizontal=st.session_state.horizontal,
    )
