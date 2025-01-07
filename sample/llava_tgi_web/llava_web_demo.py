import base64
import imghdr
import logging
import os
import sys

import streamlit as st
from text_generation import Client

if len(sys.argv) > 1:
    server_url = sys.argv[1]
else:
    server_url = "http://localhost:8099"

client = Client(server_url, timeout=60)

logging.basicConfig(
    filename="llava.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

# SYS_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "

def get_pic():
    img_type = {
        'jpg': 'image/jpg',
        'jpeg': 'image/jpeg',
        'png': 'image/png'
    }.get(st.session_state.file_extension)
    logging.info(f"Image type: {img_type}")
    return f"data:{img_type};base64,{st.session_state.base64_img.decode('utf-8')}"


def test_client_llava(question, max_new_tokens):
    try:
        concat_prompt = f"User:![]({get_pic()}){question}"
        # concat_prompt = SYS_PROMPT + f"USER: <image>\n{question}  ASSISTANT:"
        for response in client.generate_stream(concat_prompt,
                                               max_new_tokens=max_new_tokens):
            if not response.token.special:
                yield response.token.text
    except Exception as e:
        logging.error(f"Error in test_client_llava: {e}")
        return "An error occurred while processing your request."


def main():
    st.title("LLaVA Web Demo")

    if "string_data" not in st.session_state:
        st.session_state.string_data = None
    if "base64_img" not in st.session_state:
        st.session_state.base64_img = None
    if "history" not in st.session_state:
        st.session_state.history = []

    st.sidebar.header("Server Configuration")
    with st.sidebar.form("my-form", clear_on_submit=True):
        max_new_tokens = st.sidebar.slider("Max New Tokens", 10, 300, 100)
        uploaded_img = st.file_uploader("Upload an image", type=ALLOWED_EXTENSIONS)

        if uploaded_img is not None:
            st.session_state.string_data = uploaded_img.read()
            st.session_state.file_extension = imghdr.what(None, h=st.session_state.string_data)
            logging.info(f"Uploaded file extension: {st.session_state.file_extension}")
            if st.session_state.file_extension not in ALLOWED_EXTENSIONS:
                st.error("Unsupported file type!")
                return
            st.session_state.base64_img = base64.b64encode(st.session_state.string_data)

        submitted = st.form_submit_button("submit")

    if submitted:
        st.session_state.history.clear()

    image_container = st.container()
    with image_container:
        if st.session_state.string_data is not None:
            st.image(st.session_state.string_data, caption="Uploaded Image")

    messages = st.container()
    with messages:
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    text_container = st.container()
    with text_container:
        if question := st.chat_input(
            "Enter your question about the image", key="text_input"
        ):
            if st.session_state.base64_img is None:
                st.error("Please upload an image first!")
            else:
                try:
                    with messages:
                        with st.chat_message("user"):
                            st.markdown(question)
                        st.session_state.history.append({"role": "user", "content": question})
                        logging.info(f"User: {question}")
                        with st.chat_message("assistant"):
                            generated_text = st.write_stream(test_client_llava(question, max_new_tokens))
                        logging.info(f"LLaVA: {generated_text}")
                        st.session_state.history.append(
                            {"role": "assistant", "content": generated_text}
                        )
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
