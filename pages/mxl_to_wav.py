import streamlit as st
import pandas as pd
import mxl2wav as mw
import os
from scipy.io import wavfile

st.markdown("# Add widgets to sidebar")
st.sidebar.markdown("# Add widgets to sidebar")


mxlfile = st.file_uploader("Upload a mxl file")
instrument = st.selectbox(
    'How would you like to convert to ?',
    ('piano', 'violin', 'flute', 'clarinet'))

st.write('You selected:', instrument)

if mxlfile:
    if st.button("Convert to wav"):

        ans = mw.m2w(mxlfile, str(instrument))
        ans = mw.m2w(mxlfile, str(instrument))
        # sr, x = wavfile.read('pages/Data/temp.wav')
        audio_file = open('pages/Data/temp.wav', 'rb')
        audio_bytes = audio_file.read()

        st.download_button(
            label="Download data",
            data=audio_bytes,
            file_name='audio.wav'
        )
        os.remove('temp_midi.mid')
        os.remove('temp_mxl.mxl')
        os.remove('pages/Data/temp.wav')

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
