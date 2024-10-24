from openai import OpenAI

base_url = "http://127.0.0.1:18080/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)
messages = [
    {"role": "user", "content": "请用python写一个冒泡排序"}
]

response = client.chat.completions.create(
    model="qwen",
    messages=messages,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, flush=True, end='')

        