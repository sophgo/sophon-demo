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
from blip_server_itm import BlipServer
import time

app = Flask(__name__)

class Call_Blip_server:
    def __init__(self):
        self.install_dependencies()
        self.blip_server = BlipServer()

    def install_dependencies(self):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ftfy","regex","numpy"]) # 检查依赖库
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")


    def push_data(self, id, image, texts):
        data = {
            "id":id,
            "image":image,
            "texts":texts
        }
        self.blip_server.push_image(data)
        print("push_data:",data)


    def get_result(self):
        result = self.blip_server.get_batch_result()
        return result

@app.route('/push_data', methods=['POST'])
def call_blip_server_push_data():

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

        blip_server_instance.push_data(id, image_np, texts)
        push_data_time = time.time() - start_time
        print("push_data_time:",push_data_time)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_result', methods=['GET'])
def call_blip_server_get_result():
    id = request.args.get('id')
    result = blip_server_instance.get_result()
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
    blip_server_instance = Call_Blip_server() # bmodel加载、queue初始化
    app.run(host=args.host, port=args.port, debug=True) 
