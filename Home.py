import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime

today = datetime.today().strftime("%H:%M:%S")

st.title(today)

st.write("Hello")
a = [1,2,3,4]
d = {"x": 1}
p = PromptTemplate.from_template("xxxx")

a
d

st.selectbox("Choose your model", ("GPT-3", "GPT-4"))

value = st.slider('temperature', min_value=0.1, max_value=1.0)
st.write(value)