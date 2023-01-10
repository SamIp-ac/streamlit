import streamlit as st
import pitchDetection as pD
import librosa
import numpy as np
import soundfile as sf
st.markdown("# Detection whole song pitch")
st.sidebar.markdown("# Detection whole song pitch")
st.sidebar.markdown("### - upload your wav/mp3 file. The detector would help your to generate the pitch and duration.")


wavfile_ = st.file_uploader("Upload a audio wav/mp3 file")
sampling_rate = st.selectbox(
    'which is your audio sampling rate ?',
    ('44100', '22050'))

sampling_rate = int(sampling_rate)

col1, col2 = st.columns(2)

with col1:
    int_s = st.text_input('Cutting start point : ', '0')
with col2:
    int_e = st.text_input('Cutting end point : ', '-1')

int_s = int(int_s)
int_e = int(int_e)

if wavfile_:
    st.audio(wavfile_)

if st.button("Detect it"):

    if str(wavfile_.name)[-3:] == 'mp3':
        y, sr = sf.read(wavfile_, dtype=np.int16, start=int_s, stop=int_e, always_2d=True)
        if len(y[0]) == 2:
            temp = (y[:, 0] + y[:, 1])/2

        else:
            temp = y

        y = temp
        print(y)

    else:
        y, sr = librosa.load(wavfile_, sr=sampling_rate, mono=True)
        y = y[int_s:int_e]

    whole_predictionNote, whole_predictionDuration, whole_predictionError, ffL, df =\
        pD.whole_pitch(y, sampling_rate_=sampling_rate)

    st.dataframe(df)
