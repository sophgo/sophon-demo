from typing import List, Optional, Any
from transformers import AutoTokenizer
import sophon.sail as sail
import argparse
import time

from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM

from utils.qwen import Qwen

# 自定义分割器
class CustomTextSplitter(RecursiveCharacterTextSplitter):
    def split_text(self, text):
        chunks = super().split_text(text)
        # 确保句号不会出现在下一个 chunk 的开头
        for i in range(1, len(chunks)):
            if chunks[i].startswith('。'):
                chunks[i-1] += '。'
                chunks[i] = chunks[i][1:]
        return chunks


class CustomLLM(LLM):
    client: Qwen

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        callbacks: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        with open('prompt.txt', 'r', encoding='utf-8') as file:
            system_prompt = file.read()
        messages = [
            {"role": "system", "content": f"你是一个会议纪要助手，你的作用是将一段会议文本总结为会议摘要，会议摘要的格式为:\n {system_prompt} \n"},
            {"role": "user", "content": prompt}
            # 如果需要，可以在这里添加更多的消息历史
        ]

        response = self.qwen_completion(messages)
        return response
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def get_num_tokens(self, text: str) -> int:
        return len(self.client.get_token_id(text))
    
    def qwen_completion(self, messages):
        result = ""
        for response in self.client.chat_stream(messages):
            result += response
        return result



def summary(llm, text_path):
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    with open('prompt.txt', 'r', encoding='utf-8') as file:
        prompt = file.read()

    # 定义生成摘要的Prompt模板
    prompt_template = """请以以下特定的会议纪要格式总结会议内容，如果找不到相关内容可以填未提及或者不填写：

"""+ prompt + """

会议内容如下:

{text}

请总结：
"""

    text_splitter = CustomTextSplitter(separators=["\n\n", "\n", "\n\n\n"], chunk_size=8300, chunk_overlap=0)
    docs = text_splitter.create_documents([text])

    print(f"the len of text is: {len(text)}")
    print(f"you now have {len(docs)} docs intead of 1 piece of text")


    summary_prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

    # 逐段生成摘要
    summaries = []
    for doc in docs:
        chain = summary_prompt | llm
        summarie =  chain.invoke({"text": doc.page_content})
        print(f"summary part: \n{summarie}")
        summaries.append(summarie)

    output =""

    if len(summaries) == 1:
        output = summaries[0]
    else:
        final_prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
        summaries_text = ""
        for i in range(len(summaries)):
            summaries_text += f"{summaries[i]}"

        chain = final_prompt | llm
        output = chain.invoke({"text": summaries_text})
    return output

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='../models/qwen1.5-7b_int4_6k_1dev.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--text_path', type=str, default='./meeting.txt', help='meeting text file path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    start = time.time()
    args = argsparser()
    tokenizer_path = "./utils/token_config"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    client = Qwen(args.bmodel, args.dev_id, tokenizer_path)
    llm = CustomLLM(client=client)
    output = summary(llm, args.text_path)
    end = time.time()
    # 以追加模式打开文件，如果文件不存在则创建文件
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(output)
    print("summary has been saved to output.txt")
    print(f"cost time: {end - start}")
    