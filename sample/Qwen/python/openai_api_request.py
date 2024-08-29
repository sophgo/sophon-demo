from openai import OpenAI

base_url = "http://127.0.0.1:18080/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)
messages = [
    {"role": "user", "content": "你好"}
]

print(client.models.list())
response = client.chat.completions.create(
    model="qwen1.5",
    messages=messages,
    stream=False
)

# stream chat
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, flush=True, end='')

# not stream chat
# print(response.choices[0].message.content)
        