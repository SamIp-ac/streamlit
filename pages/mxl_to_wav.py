import streamlit as st
import mxl2wav as mw
import os
from scipy.io import wavfile

st.markdown("# Transform mxl file to audio")
st.sidebar.markdown("# mxl to wav convertor")


mxlfile = st.file_uploader("Upload a mxl file")
instrument = st.selectbox(
    'which instrument would you like to convert to ?',
    ('---', 'piano', 'violin', 'flute', 'clarinet'))

if instrument == '---':
    st.text('Please select one instrument')


if instrument != '---':
    st.write('You selected:', instrument)

    if st.button("Convert to wav"):
        ans = mw.m2w(mxlfile, str(instrument))

        # sr, x = wavfile.read('pages/Data/temp.wav')
        audio_file = open('pages/Data/temp_mxl2wav.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio('pages/Data/temp_mxl2wav.wav')

        os.remove('temp_midi.mid')
        os.remove('temp_mxl.mxl')
        os.remove('pages/Data/temp_mxl2wav.wav')

        st.download_button(
            label="Download data",
            data=audio_bytes,
            file_name='audio.mp3'
        )

'''# Just add it after st.sidebar:
a = st.sidebar.radio('Select one:', [1, 2])

# Or use "with" notation
with st.sidebar:
    st.radio('Select one:', [1, 2])'''
