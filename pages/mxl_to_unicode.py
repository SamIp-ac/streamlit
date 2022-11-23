import streamlit as st
import mxl2unicode as mu
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

st.markdown("# Display text")
st.sidebar.markdown("# Display text")

st.text('Fixed width text')
st.markdown('_Markdown_') # see *
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')

# displays a file uploader widget
audio_file = st.file_uploader("Upload a mxl file")

if st.button("Convert to unicode"):

    ans = mu.mxl2uni(audio_file)
    st.text(ans)

    st.download_button(
        label="Download data as txt",
        data=ans,
        file_name='unicode.txt'
    )

    copy_button = Button(label="Copy text")
    copy_button.js_on_event("button_click", CustomJS(args=ans, code="""
        navigator.clipboard.writeText(ans);
        """))

    no_event = streamlit_bokeh_events(
        copy_button,
        events="GET_TEXT",
        key="get_text",
        refresh_on_update=True,
        override_height=75,
        debounce_time=0)
