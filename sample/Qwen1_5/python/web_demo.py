#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import streamlit as st
import sophon.sail as sail
from qwen1_5 import Qwen1_5
from transformers import AutoTokenizer


token_path = './python/token_config'
bmodel_path = './models/BM1684X/qwen1.5-1.8b_int4_1dev.bmodel'
dev_id = 0

st.title("Qwen1.5-1.8B")

@st.cache_resource
def get_handle():
    return sail.Handle(dev_id)

@st.cache_resource
def get_engine():
    return sail.EngineLLM(bmodel_path, [dev_id])

@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize sail.Handle
if "handle" not in st.session_state:
    st.session_state.handle = get_handle()
# Initialize sail.Engine
if "engine" not in st.session_state:
    st.session_state.engine = get_engine()

# Initialize Tokenizer
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = get_tokenizer()

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

    client = Qwen1_5(st.session_state.handle, st.session_state.engine, st.session_state.tokenizer)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
    
        stream = client.chat_stream(input = prompt,history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])
        response = st.write_stream(stream)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})