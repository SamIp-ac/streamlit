import numpy as np
import streamlit as st
from ic_model import instruments_classify as ic
import os
from PIL import Image

st.markdown("# Predicting audio")
st.sidebar.markdown("# Predict the type of audio")


wavfile_ = st.file_uploader("Upload a audio wav file")
col1, col2 = st.columns(2)

with col1:
    int_s = st.text_input('Cutting start point : ', '0')
with col2:
    int_e = st.text_input('Cutting end point : ', '-1')
int_s = int(int_s)
int_e = int(int_e)

if wavfile_:
    st.audio(wavfile_)
    if st.button("Predict"):

        ans = ic.inst_classifier(filename=wavfile_, cutting_start=int_s, cutting_end=int_e)

        st.text('The top 5 predict is : \n')
        st.text(str([x for x in ans]))
        image = Image.open('pages/Data/temp.png')

        st.image(image, caption='Prediction result')

        st.download_button(
            label="Download image",
            data=open('pages/Data/temp.png', 'rb').read(),
            file_name='pages/Data/temp.png'
        )
        os.remove('pages/Data/temp.png')
