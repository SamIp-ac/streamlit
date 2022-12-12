import streamlit as st
from ic_model import instruments_classify as ic
import os
from PIL import Image

st.markdown("# Predicting audio")
st.sidebar.markdown("# Predict the type of audio")


wavfile_ = st.file_uploader("Upload a audio wav file")

if wavfile_:
    if st.button("Predict"):

        ans = ic.inst_classifier(filename=wavfile_)
        image = Image.open('pages/Data/temp.png')

        st.image(image, caption='Prediction result')

        st.download_button(
            label="Download image",
            data=open('pages/Data/temp.png', 'rb').read(),
            file_name='pages/Data/temp.png'
        )
        st.text('The top 5 predict is : \n')
        st.text(str([x for x in ans]))
        os.remove('pages/Data/temp.png')
