import streamlit as st
import pitchDetection as pD
import librosa

st.markdown("# Detection whole song pitch")
st.sidebar.markdown("# Detection whole song pitch")


wavfile_ = st.file_uploader("Upload a audio wav/mp3 file")

sampling_rate = st.selectbox(
    'which is your audio sampling rate ?',
    ('44100', '22050'))

sampling_rate = int(sampling_rate)

st.audio(wavfile_)

if st.button("Detect it"):

    y, sr = librosa.load('streamlit_data_app_course_Sonata_in_G.mp3', sr=sampling_rate)
    whole_predictionNote, whole_predictionDuration, whole_predictionError, ffL, df = \
        pD.whole_pitch(y, sampling_rate_=sampling_rate)

    st.dataframe(df)
