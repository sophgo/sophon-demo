import time

from flask import make_response
from flask import Flask, request, jsonify
from flask_cors import CORS
from sam_opencv import *
from sam_encoder import *

app = Flask(__name__)
CORS(app)  # 启用CORS，允许所有来源的请求


# 图片保存在服务器上的这个目录下
IMAGES_DIR = './web_ui/images'

@app.route('/favicon.ico')
def favicon():
    return ('', 204)


@app.route('/box-coordinates', methods=['POST'])
def box_coordinates():
    try:
        # 获取前端传递的JSON数据
        data = request.get_json()
        start = data['start']
        end = data['end']
        args.input_point = str(data['start']['x']) +','+ str(data['start']['y']) +','+  str(data['end']['x']) + ',' + str(data['end']['y'])
        print('Parsed Coordinates for box:  ', args.input_point)
        # 打印坐标到控制台
        src_img, results= imageProcess(args,sam_encoder_global)
        saveImages(args,src_img,results)
        mask_list = results.tolist()
        return jsonify({'maskList': mask_list})

    except Exception as e:
        # 如果发生异常，返回错误信息
        response = {
            'success': False,
            'message': str(e)
        }
        return jsonify(response)


@app.route('/images', methods=['GET'])
def list_images():
    # 列出所有的jpg文件
    images = [img for img in os.listdir(IMAGES_DIR) if img.endswith('.jpg')]
    print(images)
    return jsonify(images)


# @app.route('/coordinates/<int:coordinate_id>', methods=['OPTIONS'])
@app.route('/*', methods=['OPTIONS'])
def handle_options_request(coordinate_id):
    # 在这个函数中处理OPTIONS请求
    print('调用的OPTION 方法，不做处理，直接跳过')
    response = make_response()
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


def stop_process():
    global stop_flag
    stop_flag = True


@app.route('/stop', methods=['POST'])
def stop_process_route():
    # 接收到停止指令时调用停止函数
    data = request.get_json()
    print('数据stop处理的json', data)
    stop_process()
    return jsonify({'message': 'Stop command received'})


@app.route('/start', methods=['POST'])
def start_process_route():
    try:
        # 获取前端传递的JSON数据
        data = request.get_json()
        imageName = data['imageName']
        args.input_image = './web_ui/images/' + imageName
        # 接收到停止指令时调用开始函数
        global sam_encoder_global
        sam_encoder_global = samEncoderSave(args)
        return jsonify({'message': 'start command received'})

    except Exception as e:
        # 如果发生异常，返回错误信息
        response = {
            'success': False,
            'message': str(e)
        }
        return jsonify(response)


@app.route('/coordinates/<int:coordinate_id>', methods=['POST'])
def parse_coordinates(coordinate_id):
    try:
        # 获取前端传递的JSON数据
        data = request.get_json()
        print('数据处理的json', data)
        # 提取x和y的值
        x = int(data['x'])
        y = int(data['y'])

        # 打印x和y的值到控制台
        print(f'Parsed Coordinates for ID {coordinate_id}: x={x}, y={y}')
        args.input_point = '{0},{1}'.format(int(x),int(y))
        src_img, results= imageProcess(args,sam_encoder_global)
        saveImages(args,src_img,results)
        mask_list = results.tolist()
        return jsonify({'maskList': mask_list})

    except Exception as e:
        # 如果发生异常，返回错误信息
        response = {
            'success': False,
            'message': str(e)
        }
        return jsonify(response)


def samEncoderSave(args):
    sam_encoder = SamEncoder(args)
    return sam_encoder


def imageProcess(args,sam_encoder):
    sam_vit_b = SAM_b(args)
    batch_size = sam_vit_b.batch_size
    sam_vit_b.init()
    src_img = cv2.imread(args.input_image)
    if src_img is None:
        logging.error("{} imread is None.".format(args.input_image))
    # 初始化SAM的embedding bmodel做预处理
    sam = Sam()
    # 处理图片
    results = sam_vit_b(src_img, sam_encoder, sam)
    return  src_img,results


def saveImages(args,src_img,results):
    input_point = np.array([list(map(int, args.input_point.split(',')))])
    if len(input_point[0]) == 2:
        save_image_point(src_img,results,input_point, box = False)
    else:
        save_image_point(src_img,results,input_point, box = True)


if __name__ == '__main__':
    args = argsparser()
    # 运行Flask应用在本地的端口8000
    app.run(host='localhost', port=8000)
