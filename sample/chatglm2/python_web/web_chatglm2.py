import time
import gradio as gr
import mdtex2html
import ctypes
import argparse

class TokenWord(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_int),
        ("word", ctypes.c_char * 2048)  # 假设最大长度为 100，你可以根据实际情况调整
    ]


class TPUChatglm:
    def __init__(self,args):
        self.lib = ctypes.cdll.LoadLibrary('build/libChatGLM2.so')
        self.device_id = args.dev_id
        self.bmodel_path = args.bmodel
        self.token_path = args.token
        self.libset()
        self.init()

    def libset(self):
        self.lib.ChatGLM2_with_devid_and_model.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.ChatGLM2_with_devid_and_model.restype = ctypes.c_void_p

        self.lib.ChatGLM2_delete.argtypes = [ctypes.c_void_p]

        # deinit
        self.lib.ChatGLM2_deinit.argtypes = [ctypes.c_void_p]

        # ChatGLM2_predict_first_token
        self.lib.ChatGLM2_predict_first_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.ChatGLM2_predict_first_token.restype = ctypes.c_char_p

        # ChatGLM2_predict_next_token
        self.lib.ChatGLM2_predict_next_token.argtypes = [ctypes.c_void_p]
        self.lib.ChatGLM2_predict_next_token.restype = ctypes.c_char_p

        # get_eos
        self.lib.get_eos.argtypes = [ctypes.c_void_p]
        self.lib.get_eos.restype = ctypes.c_int
        # get_history
        self.lib.get_history.argtypes = [ctypes.c_void_p]
        self.lib.get_history.restype = ctypes.c_char_p
        # set history
        self.lib.set_history.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    def init(self):
        self.obj = self.lib.ChatGLM2_with_devid_and_model(self.device_id, self.bmodel_path.encode('utf-8'),
                                                          self.token_path.encode('utf-8'))

    def predict_first_token(self, context):
        return self.lib.ChatGLM2_predict_first_token(self.obj, context.encode('utf-8')).decode('utf-8')

    def predict_next_token(self):
        return self.lib.ChatGLM2_predict_next_token(self.obj).decode('utf-8')

    def predict(self, context):

        first_token = self.predict_first_token(context)
        # print(first_token, end='')
        res = ''
        while True:
            next_token = self.predict_next_token()
            if next_token == '_GETMAX_' or next_token == '_GETEOS_':
                # print(next_token)
                break
            # print(next_token, end='')
            res += next_token
        return res

    def stream_predict(self, query, history):
        history.append((query, ''))

        prompt = ''
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        
        res = ''
        first_token = self.predict_first_token(prompt)
        res += first_token

        while True:
            next_token = self.predict_next_token()
            if next_token == '_GETMAX_' or next_token == '_GETEOS_':
                break
            res += next_token
            history[-1] = (query, res)
            yield res, history

    def get_config(self):
        pass


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def gen(input, history):
    i = 0
    history.append((input, ''))
    res = ''
    while i < 10:
        i += 1
        res += str(i)
        time.sleep(0.05)
        history[-1] = (input, res)
        yield res, history


def predict(input, chatbot, max_length, top_p, temperature, history):

    chatbot.append((parse_text(input), ""))
    for response, history in glm.stream_predict(input, history):
        chatbot[-1] = (parse_text(input), parse_text(response))
        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None



if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='models/BM1684X/chatglm2-6b.bmodel', help='path of token')
    parser.add_argument('--token', type=str, default='models/BM1684X/tokenizer.model', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()


    gr.Chatbot.postprocess = postprocess
    glm = TPUChatglm(args)
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">ChatGLM2-6B TPU</h1>""")

        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])

        submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history],
                        [chatbot, history], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    demo.queue().launch(share=True, server_name="0.0.0.0", inbrowser=True)
