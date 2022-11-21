import streamlit as st

st.markdown("# Play Audio")
st.sidebar.markdown("# Play Audio️")

st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Rhythm detection")

# displays a file uploader widget
audio_file = st.file_uploader("Upload a audio (.wav)")

if st.button("Style Transfer"):
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)