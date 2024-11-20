from openai import OpenAI
import argparse

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('--stream', action='store_true', help='use stream chat or not (default: False)')
args = parser.parse_args()

base_url = "http://127.0.0.1:18080/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)
messages = [
    {"role": "user", "content": "你好"}
]

print(client.models.list())
response = client.chat.completions.create(
    model="qwen2.5",
    messages=messages,
    stream=args.stream
)

if args.stream:
    # stream chat
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, flush=True, end='')
    print('')
else:
    # not stream chat
    print(response.choices[0].message.content)