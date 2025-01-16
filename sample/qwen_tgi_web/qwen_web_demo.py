import sys

import streamlit as st
from openai import OpenAI

if len(sys.argv) > 1:
    server_url = sys.argv[1]
else:
    server_url = "http://localhost:8090/v1"


st.title("Qwen2-7B-Instruct")


def get_client():
    return OpenAI(api_key="api_key", base_url=server_url)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


if "client" not in st.session_state:
    st.session_state.client = get_client()
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("请输入您的问题 "):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        stream = st.session_state.client.chat.completions.create(
            model="qwen2-72b-instruct",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            max_tokens=1024,
        )
        response = st.write_stream(stream)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
