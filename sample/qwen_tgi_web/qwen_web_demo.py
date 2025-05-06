import sys
import time

import streamlit as st
import tiktoken  # Lightweight token counting library
from openai import OpenAI

# Configuration parameters
MAX_CONTEXT_TOKENS = 2048  # Hard limit of 2048 tokens
MODEL_NAME = "qwen2-72b-instruct"  # Keep consistent with TGI service
TOKEN_OVERHEAD_PER_MESSAGE = 3  # Format overhead per message
TOKEN_BUFFER_VALUE = 50 # Token buffer value

if len(sys.argv) > 1:
    server_url = sys.argv[1]
    if len(sys.argv) > 2:
        MAX_CONTEXT_TOKENS = int(sys.argv[2])
else:
    server_url = "http://localhost:8090/v1"

st.title(f"{MODEL_NAME} ")

# Initialize token encoder
try:
    enc = tiktoken.get_encoding("cl100k_base")  # Universal encoding scheme
except Exception as e:
    st.error(f"Token编码器初始化失败: {str(e)}")
    st.stop()


def count_tokens(text):
    """Precisely calculate the number of tokens in text"""
    return len(enc.encode(text))


def calculate_total_tokens(messages):
    """Calculate the total number of tokens in the entire conversation context"""
    total = 0
    for msg in messages:
        total += count_tokens(msg["content"]) + TOKEN_OVERHEAD_PER_MESSAGE
    return total


# Initialize client and chat history
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key="api_key", base_url=server_url)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.token_count = 0  # New token counter

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input processing
if prompt := st.chat_input("请输入您的问题"):
    # Pre-check token limit
    new_user_tokens = count_tokens(prompt) + TOKEN_OVERHEAD_PER_MESSAGE
    if st.session_state.token_count + new_user_tokens > MAX_CONTEXT_TOKENS:
        st.session_state.messages = []
        st.session_state.token_count = 0
        st.rerun()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.token_count += new_user_tokens

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response (with real-time token monitoring)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response_tokens = 0

        try:
            # Calculate remaining available tokens
            remaining_tokens = (
                MAX_CONTEXT_TOKENS - st.session_state.token_count - TOKEN_BUFFER_VALUE
            )  # Reserve 50 token buffer

            stream = st.session_state.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
                max_tokens=1024,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    full_response += chunk_content
                    message_placeholder.markdown(full_response + "▌")

                    # Update token count in real-time
                    response_tokens = count_tokens(full_response)
                    current_total = (
                        st.session_state.token_count
                        + response_tokens
                        + TOKEN_OVERHEAD_PER_MESSAGE
                    )

                    # Hard truncation protection
                    if current_total >= MAX_CONTEXT_TOKENS - 10:
                        full_response += (
                            "\n\n[系统] 已达到token限制，3秒后将自动刷新..."
                        )
                        message_placeholder.markdown(full_response)
                        time.sleep(3)  # Show prompt for 3 seconds
                        st.session_state.messages = []
                        st.session_state.token_count = 0
                        st.rerun()  # Refresh page
                        break

            # Finally add assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            st.session_state.token_count += response_tokens + TOKEN_OVERHEAD_PER_MESSAGE
            message_placeholder.markdown(full_response)

            # Final check if limit is exceeded
            if st.session_state.token_count > MAX_CONTEXT_TOKENS:
                st.session_state.messages = []
                st.session_state.token_count = 0
                st.rerun()

        except Exception as e:
            st.error(f"生成失败: {str(e)}")
            st.session_state.messages.pop()  # Rollback user message
            st.session_state.token_count -= new_user_tokens
