import streamlit as st
import requests
import base64
import time
import random
from PIL import Image
from io import BytesIO
import pandas as pd
import sys

if len(sys.argv) > 1:
    server_url = sys.argv[1]
else:
    server_url = 'http://localhost:8080'

st.title("Groundedsam Server Frontend")

# 创建一个全局字典来存储历史记录
if 'history' not in st.session_state:
    st.session_state.history = {}
def update_selected_id():
    st.session_state.selected_id = st.session_state["temp_selected_id"]


tab1, tab2 = st.tabs(["当前上传", "历史记录"])
with tab1:
    with st.sidebar.form("my-form", clear_on_submit=True):
        # 上传多个图片文件
        uploaded_files = st.file_uploader("选择图片", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        text_prompt = st.text_input("输入Prompt")
        text_threshold = st.number_input("text_threshold, default=0.20", min_value=0.0, max_value=1.0, value=0.2)
        box_threshold = st.number_input("box_threshold, default=0.30", min_value=0.0, max_value=1.0, value=0.3)
        submitted = st.form_submit_button("UPLOAD!")

    if submitted and uploaded_files is not None:
        st.write("进行推理")

        for uploaded_file in uploaded_files:
            # 生成不重复的id
            id = int(time.time() * 1000) + random.randint(0, 999)

            # 显示上传的图片
            uploaded_file.seek(0)  # 重置文件指针位置
            image = Image.open(uploaded_file)
            st.image(image, caption=f'上传的图片: {uploaded_file.name}', use_column_width=True)
            
            start_time = time.time() 
            # 将图片转换为base64格式，然后发给后端，节省网络带宽
            uploaded_file.seek(0)  # 再次重置文件指针位置
            image_base64 = base64.b64encode(uploaded_file.read()).decode('utf-8')

            # 发送数据到后端
            response = requests.post(f'{server_url}/push_data', data={
                'id': id,
                'image': image_base64,
                "text_prompt": text_prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold
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
                        time.sleep(0.1)  # 等待0.1秒后重试
                st.success(f"结果获取成功，ID: {id}, 数据处理及获取结果耗时: {time.time() - submit_time:.2f}秒")
                # 使用pandas DataFrame来格式化显示结果
                result_df = pd.DataFrame({
                    'text_prompt': result['text_prompt'],
                    '预测文本': result['pred_phrases'],
                    # 'boxes_filt': result['boxes_filt']
                })
                img_data = base64.b64decode(result["pred_image"])
                img_bytes = BytesIO(img_data)
                pred_img = Image.open(img_bytes)

                # 使用 Streamlit 显示图像
                st.write("推理结果：")
                st.image(pred_img, use_column_width=True)
                st.table(result_df)

                # 添加当前上传记录到历史记录
                st.session_state.history[id] = {
                    'id': id,
                    'image': uploaded_file,
                    'filename': uploaded_file.name,
                    'result': result,
                    'pred_img': pred_img
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
                st.image(selected_record['image'], caption=f'上传的图片: {selected_record["filename"]}', use_column_width=True)
                # 使用pandas DataFrame来格式化显示结果
                result_df = pd.DataFrame({
                    'text_prompt': selected_record['result']['text_prompt'],
                    '预测文本': selected_record['result']['pred_phrases']
                })
                st.write("推理结果：")
                st.image(selected_record['pred_img'], use_column_width=True)
                
                st.table(result_df)

            except Exception as e:
                st.error(f"发生错误: {str(e)}")  # 调试信息
    else:
        st.write("没有历史记录")
