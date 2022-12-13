import streamlit as st
import pandas as pd
from auto_speech_v1_3_1 import auto_speech as ah
import os

st.markdown("# Transform mxl file to speech")
st.sidebar.markdown("# mxl to wav speech convertor")

mxlfile = st.file_uploader("Upload a mxl file")

if st.button("Convert to wav"):
    ans = ah(mxlfile)

    audio_file = open('pages/Data/temp_autospeech.wav', 'rb')
    audio_bytes = audio_file.read()

    st.audio('pages/Data/temp_autospeech.wav')
    os.remove('pages/Data/temp_autospeech.wav')

    st.download_button(
        label="Download data",
        data=audio_bytes,
        file_name='audio.mp3'
    )