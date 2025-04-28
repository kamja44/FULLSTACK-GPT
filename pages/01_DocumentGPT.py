import time
import streamlit as st

st.title("DocumentGPT")


# chat_message를 human으로 설정했기 때문에 with 아래에 작성된 모든 것들은 human이 작성한 것이 된다.
with st.chat_message("human"):
    st.write("Hello")

# 마찬가지로 모두 ai가 작성한게 된다.
with st.chat_message("ai"):
    st.write("how are you")

st.chat_input('Send a message to the ai')

# chatbot이 어떤 작업을 할 때 커뮤니케이션을 만들 수 있다.
with st.status("Embedding file...", expanded=True) as status:
    time.sleep(2)
    st.write("Getting the file")
    time.sleep(2)
    st.write("Embedding the file")
    time.sleep(2)
    st.write("Caching the file")
    status.update(label="Error", state="error")

