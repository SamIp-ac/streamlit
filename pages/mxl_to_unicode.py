import streamlit as st
import mxl2unicode as mu
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import os

st.markdown("# Unicode generator")
st.sidebar.markdown("# Unicode generator")

# displays a file uploader widget
mxl_file = st.file_uploader("Upload a mxl file")
if mxl_file:
    if st.button("Convert to unicode"):
        ans = mu.mxl2uni(mxl_file)
        st.text(ans)
        os.remove('temp.wav')
        # print(ans.encode('utf-8').decode('utf-8'))

        st.download_button(
            label="Download data as txt",
            data=ans,
            file_name='unicode.txt'
        )


        copy_button = Button(label="Copy text")
        copy_button.js_on_event("button_click", CustomJS(args=dict(df=ans), code="""
            navigator.clipboard.writeText(df);
            """))

        no_event = streamlit_bokeh_events(
            copy_button,
            events="GET_TEXT",
            key="get_text",
            refresh_on_update=True,
            override_height=25,
            debounce_time=0)
