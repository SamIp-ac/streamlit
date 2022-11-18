import streamlit as st

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")

st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Rhythm detection")

# displays a file uploader widget
audio_file = st.file_uploader("Upload a audio (.wav)")

if st.button("Style Transfer"):
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)