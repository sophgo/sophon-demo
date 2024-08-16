# -*- coding: utf-8 -*-
import time
import gradio as gr
from pipeline import Llama3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
parser.add_argument('--enable_history', type=bool, default=True, help="if set, enables storing of history memory.")
parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
parser.add_argument('--decode_mode', type=str, default="basic", choices=["basic", "jacobi"], help='mode for decoding')
args = parser.parse_args()

model = Llama3(args)


def gr_update_history():
    if model.model.token_length >= model.SEQLEN:
        # print("... (reach the maximal length)", flush=True, end='')
        gr.Warning("reach the maximal length, Llama3 would clear all history record")
        model.history = [model.system]
    else:
        model.history.append({"role": "assistant", "content": model.answer_cur})


def gr_user(user_input, history):
    model.input_str = user_input
    return "", history + [[user_input, None]]


def gr_chat(history):
    """
    Stream the prediction for the given query.
    """
    tokens = model.encode_tokens()

    # check tokens
    if not tokens:
        gr.Warning("Sorry: your question is empty!!")
        return
    if len(tokens) > model.SEQLEN:
        gr.Warning(
            "The maximum question length should be shorter than {} but we get {} instead.".format(
                model.SEQLEN, len(tokens)
            )
        )
        gr_update_history()

    model.answer_cur = ""
    model.answer_token = []
    token_num = 0

    first_start = time.time()
    token = model.model.forward_first(tokens)
    first_end = time.time()

    history[-1][1] = ""
    full_word_tokens = []

    while token not in model.EOS and model.model.token_length < model.SEQLEN:
        full_word_tokens.append(token)
        t_word = model.tokenizer.decode(full_word_tokens, skip_special_tokens=True)

        if "�" in t_word:
            token = model.model.forward_next()
            token_num += 1
            continue

        model.answer_token += full_word_tokens

        history[-1][1] += t_word
        full_word_tokens = []
        yield history
        token = model.model.forward_next()
        token_num += 1

    next_end = time.time()
    first_duration = first_end - first_start
    next_duration = next_end - first_end
    tps = token_num / next_duration

    print()
    print(f"FTL: {first_duration:.3f} s")
    print(f"TPS: {tps:.3f} token/s")

    model.answer_cur = model.tokenizer.decode(model.answer_token)
    gr_update_history()


def reset():
    model.clear()
    return [[None, None]]



description = """
# Llama3 TPU 🏁 
"""
with gr.Blocks() as demo:
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="Llama3", height=1050)

            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="Ask Llama3", lines=1, min_width=300, scale=6)
                submitBtn = gr.Button("Submit", variant="primary", scale=1)
                emptyBtn = gr.Button(value="Clear", scale=1)

    user_input.submit(gr_user, [user_input, chatbot], [user_input, chatbot]).then(gr_chat, chatbot, chatbot)
    # clear_history.
    submitBtn.click(gr_user, [user_input, chatbot], [user_input, chatbot]).then(gr_chat, chatbot, chatbot)

    emptyBtn.click(reset, outputs=[chatbot])

demo.queue(max_size=20).launch(share=False, server_name="0.0.0.0", inbrowser=True, server_port=8003)

