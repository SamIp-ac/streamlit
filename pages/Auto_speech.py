import streamlit as st
import pandas as pd

df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
st.markdown("# Display data")
st.sidebar.markdown("# Display data")

st.dataframe(df)
st.table(df.iloc[0:1])
st.json({'foo': 'bar', 'fu': 'ba'})
st.metric('My metric', 42, 2)