# streamlit
For streamlit only
The first try of streamlit with some useful api.

Such as:
mxl file convert to unicode (use in Bravura). --Without beam now.
mxl file convert to audio (.wav) file with different instrument. --4 instruments now.
rhythm detection for small music prahses. The mechanism is spectrogram--denoise--1DTransform--clustering. --Not so work now.
auto-speech function will be added be the future.

20-12-2022
- audio classification added with cutting selection

28-12-2022
- main theme change (view ./.streamlit/config.toml)

1-1-2023
- Deployed by GCP: https://streamlit-demo-amd64-vsdczlx2ia-de.a.run.app
- You use the .mxl file (provided in my GitHub)to try, .mxl link: https://github.com/SamIp-ac/streamlit/blob/main/Concertino_in_D_Op15_3rd.mxl
