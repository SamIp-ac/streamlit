import streamlit as st
import mxl2unicode as mu
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
    None
