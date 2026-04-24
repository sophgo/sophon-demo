from flask import Flask, request, jsonify, send_file
import os
import sys
import numpy as np
import base64
import cv2
import argparse

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
sys.path.append(os.path.join(script_dir, '../python'))
from gdsam_server import gdsamServer

app = Flask(__name__)

class Call_gdsam_server:
    def __init__(self):
        self.gdsam_server = gdsamServer()

    def push_data(self, id, image, text_prompt, box_threshold, text_threshold):
        data = {
            "id":id,
            "image":image,
            "text_prompt":text_prompt,
            "box_threshold":box_threshold,
            "text_threshold":text_threshold
        }
        self.gdsam_server.push_image(data)
        print("push_data:",data)

    def get_result(self):
        result = self.gdsam_server.get_batch_result()
        return result

@app.route('/push_data', methods=['POST'])
def call_gdsam_server_push_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or missing JSON data"}), 400
        image = data.get('image')
        if not image:
            return jsonify({"error": "image field is empty"}), 400

        image_np = base64.b64decode(image)
        nparr = np.frombuffer(image_np, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_np is None:
            return jsonify({"error": "cv2.imdecode failed"}), 400
        id = int(data.get('id'))
        text_prompt = data.get('text_prompt')
        box_threshold = float(data.get('box_threshold'))
        text_threshold = float(data.get('text_threshold'))
        gdsam_server_instance.push_data(id, image_np, text_prompt, box_threshold, text_threshold)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_result', methods=['GET'])
def call_gdsam_server_get_result():
    id = request.args.get('id')
    result = gdsam_server_instance.get_result()
    if result is None:
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
    gdsam_server_instance = Call_gdsam_server()
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False) 
