#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
import streamlit as st
import requests
import base64
import time
import random
from PIL import Image
import io
import pandas as pd
import sys
import os
import argparse


if len(sys.argv) > 1:
    server_url = sys.argv[1]
else:
    server_url = 'http://localhost:8080'


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
sys.path.append(script_dir)
# 设置默认图片
default_image_path = script_dir+'/../datasets/Clothes-and-hats-misidentified-as-safety-helmet.jpg'

# 设置全局图片样式
st.markdown("<style>img { height: 300px; }</style>", unsafe_allow_html=True)  # 设置图片高度为300

use_default_image = False
st.title("CLIP Server Frontend")

# 创建一个全局字典来存储历史记录
if 'history' not in st.session_state:
    st.session_state.history = {}
def update_selected_id():
    st.session_state.selected_id = st.session_state["temp_selected_id"]


tab1, tab2 = st.tabs(["当前上传", "历史记录"])
with tab1:
    with st.sidebar.form("my-form", clear_on_submit=True):
        # 上传多个图片文件
        uploaded_files = st.file_uploader("选择图片", type=["png", "jpg", "jpeg"], accept_multiple_files=True, help="支持多张图片上传，如果不上传将使用demo默认图片进行推理")
        
        # 如果没有上传文件，则使用默认图片
        if uploaded_files is None or len(uploaded_files) == 0:
            uploaded_files = [default_image_path]
            use_default_image = True

        # 动态文本输入框数量
        num_texts = st.number_input("输入文本框数量", min_value=1, max_value=10, value=2)
        texts = [st.text_input(f"输入文本 {i+1}") for i in range(num_texts)]
        submitted = st.form_submit_button("UPLOAD!")

    if submitted and uploaded_files is not None:
        st.write("进行推理：")

        for uploaded_file in uploaded_files:
            # 生成不重复的id
            id = int(time.time() * 1000) + random.randint(0, 999)

            # 显示上传的图片
            # uploaded_file.seek(0)  # 重置文件指针位置
            start_time = time.time() 
            image = Image.open(uploaded_file)
            if use_default_image:
                st.image(image, caption=f'上传的图片: {"default"}', use_column_width="auto")  
                
                with open(uploaded_file, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')
                
            else:
                st.image(image, caption=f'上传的图片: {uploaded_file.name}', use_column_width="auto")  
                uploaded_file.seek(0)  # 再次重置文件指针位置
                image_base64 = base64.b64encode(uploaded_file.read()).decode('utf-8')


            # 检查texts的值，如果为空就删掉
            texts = [text for text in texts if text]
            if texts is None or len(texts) == 0:
                st.error("请输入文本内容")

            # 发送数据到后端
            response = requests.post(f'{server_url}/push_data', data={
                'id': id,
                'texts': texts, 
                'image': image_base64
            })

            if response.status_code == 200:
                submit_time = time.time()
                st.success(f"数据提交成功，ID: {id}, 数据提交耗时: {submit_time-start_time:.2f}秒")

                # 自动获取结果
                result = None
                while result is None:
                    result_response = requests.get(f'{server_url}/get_result?id={id}')
                    if result_response.status_code == 200:
                        result = result_response.json()
                    else:
                        time.sleep(0.01)  # 等待0.1秒后重试
                st.success(f"结果获取成功，ID: {id}, 数据处理及获取结果耗时: {time.time() - submit_time:.2f}秒")
                # 使用pandas DataFrame来格式化显示结果
                result_df = pd.DataFrame({
                    '文本': result['texts'],
                    '相似度': result['similarity']
                })

                st.write("推理结果：")
                st.table(result_df)

                # 添加当前上传记录到历史记录
                if use_default_image:
                    st.session_state.history[id] = {
                        'id': id,
                        'image': image,
                        'filename': "default",
                        'result': result
                    }
                else:
                    st.session_state.history[id] = {
                        'id': id,
                        'image': uploaded_file,
                        'filename': uploaded_file.name,
                        'result': result
                    }

            else:
                st.error(f"错误: {response.json().get('error')}")

with tab2:
    if st.sidebar.button("删除历史记录"):
        st.session_state.history = {}
    st.header("历史上传记录")
    if 'history' in st.session_state and st.session_state.history:
        # 获取所有历史记录的id
        history_ids = list(st.session_state.history.keys())

        # 使用 on_change 参数和 key 参数
        selected_id = st.selectbox(
            "选择一个历史记录",
            history_ids,
            key="temp_selected_id",
            on_change=update_selected_id
        )

        if 'selected_id' in st.session_state:
            try:
                selected_record = st.session_state.history[st.session_state.selected_id]
                st.image(selected_record['image'], caption=f'上传的图片: {selected_record["filename"]}', use_column_width="auto")  
                # 使用pandas DataFrame来格式化显示结果
                result_df = pd.DataFrame({
                    '文本': selected_record['result']['texts'],
                    '相似度': selected_record['result']['similarity']
                })
                st.write("推理结果：")
                st.table(result_df)

            except Exception as e:
                st.error(f"发生错误: {str(e)}")  # 调试信息
    else:
        st.write("没有历史记录")
