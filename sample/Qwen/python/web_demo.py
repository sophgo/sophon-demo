#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import streamlit as st
from qwen import Qwen
import argparse
import yaml

# 解析命令行参数
parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('--config', type=str, default='./config/web.yaml', help='path of config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
st.title(config["title"])

def get_client():
    return Qwen(config["bmodel_path"], config["dev_ids"], config["token_path"])


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Initialize Tokenizer
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
        
        stream = st.session_state.client.chat_stream([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])
        response = st.write_stream(stream)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})