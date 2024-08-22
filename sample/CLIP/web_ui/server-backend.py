#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
from flask import Flask, request, jsonify
import subprocess
import os
import sys
import numpy as np
from PIL import Image
import base64 
import cv2
import argparse

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
sys.path.append(os.path.join(script_dir, '../python'))
from clip_server import ClipServer
import time

app = Flask(__name__)




def call_zeroshot_predict(image_path, texts, image_model, text_model, dev_id, script_path='../../sample/CLIP/python/clip_server.py'):
    try:
        # install_dependencies()  # 安装依赖库
        args = [
            sys.executable, script_path,
            '--image_path', image_path,
            '--image_model', image_model,
            '--text_model', text_model,
            '--dev_id', dev_id,
        ] + ['--text'] + texts  # 将每个文本项单独传递
        result = subprocess.run(args, capture_output=True, text=True, check=True)

        # 处理 stdout
        lines = result.stdout.splitlines()
        output = {}
        
        # 提取 text 列表内容及其预测概率值
        for text in texts:
            for line in lines:
                if f"Text: {text}," in line:
                    similarity = line.split("Similarity: ")[1]
                    output[text] = similarity
        # 提取 preprocess、image_encode、text_encode 的时间
        for line in lines:
            if "preprocess(ms)" in line:
                output["preprocess(ms)"] = line.split(": ")[1]
            elif "image_encode(ms)" in line:
                output["image_encode(ms)"] = line.split(": ")[1]
            elif "text_encode(ms)" in line:
                output["text_encode(ms)"] = line.split(": ")[1]

        return {"stdout": output, "stderr": result.stderr, "returncode": result.returncode}
    except subprocess.CalledProcessError as e:
        return {"error": str(e), "stderr": e.stderr, "returncode": e.returncode}

class Call_Clip_server:
    def __init__(self):
        self.install_dependencies()
        self.clip_server = ClipServer()

    def install_dependencies(self):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ftfy","regex","torch","torchvision","numpy"]) # 检查依赖库
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")


    def push_data(self, id, image, texts):
        data = {
            "id":id,
            "image":image,
            "texts":texts
        }
        self.clip_server.push_image(data)
        print("push_data:",data)


    def get_result(self):
        result = self.clip_server.get_batch_result()
        return result

@app.route('/push_data', methods=['POST'])
def call_clip_server_push_data():

    try:
        start_time = time.time()

        image = request.form.get('image')
        image_np = base64.b64decode(image)# 很耗时,但是可以节省网络带宽
        decode_time = time.time() - start_time
        print("decode_time:",decode_time)
        # 转成mat，以便后续的cv2操作
        nparr = np.frombuffer(image_np, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        decode_time = time.time() - start_time
        print("decode_time:",decode_time)
        # 获取其他表单数据
        id = int(request.form.get('id'))
        texts = request.form.getlist('texts')

        clip_server_instance.push_data(id, image_np, texts)
        push_data_time = time.time() - start_time
        print("push_data_time:",push_data_time)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_result', methods=['GET'])
def call_clip_server_get_result():
    id = request.args.get('id')
    result = clip_server_instance.get_result()
    if result == None:
        return jsonify({"error": "result is None,you should sent data first"}), 400

    print(result)

    return jsonify(result)


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--host', type=str, default='0.0.0.0', help='host')
    parser.add_argument('--port', type=int, default=8080, help='port')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    clip_server_instance = Call_Clip_server() # bmodel加载、queue初始化
    app.run(host=args.host, port=args.port, debug=True,use_reloader=False) 