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
from qwen_vl import Qwen
from transformers import AutoTokenizer
from PIL import Image
import configparser

config = configparser.ConfigParser()
config.read('python/config.ini')
token_path = config.get('qwenvl','token_path')
bmodel_path = config.get('qwenvl','bmodel_path')
dev_id = int(config.get('qwenvl', 'dev_id'))

st.title("Qwen-VL")

# Function to display uploaded image in the sidebar
def display_uploaded_image(image):
    st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)

# uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
with st.sidebar:
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

# Check if a file was uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display the uploaded image in the sidebar
    display_uploaded_image(uploaded_file)

    @st.cache_resource
    def get_handle():
        return sail.Handle(dev_id)

    # @st.cache_resource
    # def get_vit():
    #     return sail.Engine(vit_path, dev_id[1], sail.IOMode.DEVIO)

    @st.cache_resource
    def get_llm():
        # return sail.Engine(bmodel_path, dev_id, sail.IOMode.DEVIO)
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
    if "llm_engine" not in st.session_state:
        st.session_state.llm_engine = get_llm()

    # Initialize Tokenizer
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = get_tokenizer()

    # Initialize client
    if "client" not in st.session_state:
        st.session_state.client = Qwen(st.session_state.handle, st.session_state.llm_engine, st.session_state.tokenizer)
        st.success('模型初始化完成！欢迎您根据图片提出问题，我将会为您解答。', icon='🎉')
        st.balloons()

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
            stream = st.session_state.client.chat_stream(input=prompt, image=image, history=[[m["role"], m["content"]] for m in st.session_state.messages])
            response = st.write_stream(stream)

            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
