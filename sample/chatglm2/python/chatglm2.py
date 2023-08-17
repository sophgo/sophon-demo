import ChatGLM2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='models/BM1684X/codegeex2-6b.bmodel', help='path of token')
    parser.add_argument('--token', type=str, default='models/BM1684X/tokenizer.model', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    args = parser.parse_args()


    bmodel = args.bmodel
    token = args.token
    dev_id = args.dev_id


    engine = ChatGLM2.ChatGLM2()
    engine.init(dev_id,bmodel,token)
    engine.answer("你好！")
    engine.chat()
    engine.deinit()
