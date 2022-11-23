import streamlit as st
import rhythm_model as rm

st.markdown("# Play Audio")
st.sidebar.markdown("# Make prediction")
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Rhythm detection")

# displays a file uploader widget
audio_file = st.file_uploader("Upload a audio (.wav)", type='wav')

if audio_file:
    if st.button("play audio"):
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
    if st.button("Predict rhythm"):
        ans = rm.classifier_(audio_file)
        st.text('The separation of rhythm is : ' + str(ans))

