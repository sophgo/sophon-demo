import argparse
import os
import sys

import streamlit as st
from app.utils.logger import setup_logging

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', '-c',
        default='./app/config/config.json',
        help='Path to the configuration file.'
    )
    args = parser.parse_args()
    return args

args = parse()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_ENGINE_CONFIG_PATH = os.path.realpath(args.config)
RAG_LOGGER_CONFIG_PATH = os.path.join(BASE_DIR, "app/config/logging.json")
sys.path.append(BASE_DIR)

setup_logging(RAG_LOGGER_CONFIG_PATH)

from app.engine.config import RAGConfig
from app.engine.rag_engine import RAGEngine


def query(prompt: str):
    result = st.session_state.rag_engine.query(prompt)
    if result["answer"]:
        st.write(result["answer"])
        if result["reference"]:
            with st.expander("参考文档"):
                for ref in result["reference"]:
                    st.info(
                        f"参考文件名: {ref[0][st.session_state.prefix_len:]}"
                    )  # remove /tmp/
                    st.write(f"相关内容: {ref[1]}")
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
    else:
        st.error("无法生成答案，请稍后重试。")


def query_stream(prompt: str):
    result_stream = st.session_state.rag_engine.query_stream(prompt)
    answer_placeholder = st.empty()
    full_answer = ""
    try:
        for partial_result in result_stream:
            if partial_result["answer"] is not None:
                full_answer += partial_result["answer"]
                answer_placeholder.write(full_answer)
            else:
                st.error("无法生成答案，请稍后重试。")
                break

        if full_answer and partial_result["reference"]:
            with st.expander("参考文档"):
                for ref in partial_result["reference"]:
                    st.info(
                        f"参考文件名: {ref[0][st.session_state.prefix_len:]}"
                    )  # remove /tmp/
                    st.write(f"相关内容: {ref[1]}")
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
    except Exception as e:
        st.error(f"生成答案时发生错误: {e}")


if __name__ == "__main__":

    st.title("RAG Demo")

    if "rag_engine" not in st.session_state:
        st.session_state.rag_config = RAGConfig.from_json(RAG_ENGINE_CONFIG_PATH)
        st.session_state.rag_engine = RAGEngine(st.session_state.rag_config)
        st.session_state.support_docx = (
            st.session_state.rag_engine.check_if_support_docx()
        )
        if st.session_state.support_docx:
            st.session_state.file_uploader_str = (
                "选择一个文档（支持 PDF、DOCX、TXT 等格式）"
            )
            st.session_state.file_uploader_list = ["pdf", "docx", "txt", "md"]
        else:
            st.session_state.file_uploader_str = "选择一个文档（支持 PDF、TXT 等格式）"
            st.session_state.file_uploader_list = ["pdf", "txt", "md"]
        st.session_state.query_func = (
            query_stream
            if st.session_state.rag_engine.check_query_stream_support()
            else query
        )
        if os.name == "nt":
            st.session_state.prefix = "D:/"
        elif os.name == "posix":
            st.session_state.prefix = "/tmp/"

        st.session_state.prefix_len = len(st.session_state.prefix)

    st.header("选择文档")
    uploaded_file = st.file_uploader(
        st.session_state.file_uploader_str,
        st.session_state.file_uploader_list,
        accept_multiple_files=True,
    )
    submitted = st.button("上传")
    if uploaded_file is not None and submitted:
        for file in uploaded_file:
            file_path = f"{st.session_state.prefix}{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if st.session_state.rag_engine.add_doc(file_path):
                st.success(f"文档 '{file.name}' 添加成功！")
            else:
                st.error(f"文档 '{file.name}' 添加失败！")

    st.header("问答")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("请输入您的问题："):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.session_state.query_func(prompt)

    with st.sidebar:
        # st.header("系统状态")
        # status = st.session_state.rag_engine.get_status()
        # st.json(status)

        st.header("删除文档")
        document_list = st.session_state.rag_engine.get_doc_list()
        if document_list:
            doc_list_remove_prefix = [
                doc[st.session_state.prefix_len :] for doc in document_list
            ]  # remove /tmp/
            selected_document = st.selectbox("选择要删除的文档", doc_list_remove_prefix)
            if st.button("删除"):
                if st.session_state.rag_engine.remove_doc(
                    st.session_state.prefix + selected_document
                ):
                    st.success(f"文档 '{selected_document}' 删除成功！")
                else:
                    st.error(f"文档 '{selected_document}' 删除失败！")
        else:
            st.info("当前没有加载的文档。")

        st.header("参数配置")
        topk = st.number_input(
            "设置最大参考数量",
            min_value=1,
            max_value=128,
            value=st.session_state.rag_config.doc_config.topk,
            step=1,
        )
        if topk:
            st.session_state.rag_engine.update_topk(topk)
