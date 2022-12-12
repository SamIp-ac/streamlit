import streamlit as st
from ic_model import instruments_classify as ic
import os
from PIL import Image

st.markdown("# Predicting instruments")
st.sidebar.markdown("# Predict instruments")


wavfile_ = st.file_uploader("Upload a audio wav file")

if wavfile_:
    if st.button("Predict"):

        ans = ic.inst_classifier(filename=wavfile_)
        image = Image.open('pages/Data/temp.png')

        st.image(image, caption='Prediction result')

        st.download_button(
            label="Download data",
            data=image,
            file_name='image.png'
        )
        os.remove('pages/Data/temp.png')
