import time
import os
import numpy as np
import argparse
import json
import sophon.sail as sail
import math
import cv2

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def inference(bmodel_path, input_path, tpu_id):
    """ Load a bmodel and do inference.
    Args:
     bmodel_path: Path to bmodel
     input_path: Path to input directory
     tpu_id: ID of TPU to use
     results_path: Path of result file

    Returns:
        True for success and False for failure
    """
    # init hyperparams
    max_video_length = 300
    step = 6
    # init Engine
    engine = sail.Engine(tpu_id)
    # load bmodel without built in input and output tensors
    engine.load(bmodel_path)
    # get model info
    # only one model loaded for this engine
    # only one input tensor and only one output tensor in this graph
    graph_name = engine.get_graph_names()[0]
    input_name = engine.get_input_names(graph_name)[0]
    output_name = engine.get_output_names(graph_name)[0]
    input_shape = engine.get_input_shape(graph_name, input_name)
    input_shapes = {input_name: input_shape}
    batch_size = input_shape[0]
    frame_size = input_shape[2]
    output_shape = engine.get_output_shape(graph_name, output_name)
    input_dtype= engine.get_input_dtype(graph_name, input_name)
    output_dtype = engine.get_output_dtype(graph_name, output_name)
    handle = engine.get_handle()
    output = sail.Tensor(handle, output_shape, output_dtype, True, True)
    output_tensors = {output_name: output}
    # get handle to create input and output tensors

    # set io_mode
    engine.set_io_mode(graph_name, sail.IOMode.SYSIO)
    
    if os.path.isdir(input_path):
        input_directory = os.listdir(input_path)
        input_directory.sort(key = lambda x: x)

    # init bmimages
    correct_count = 0
    total_count = 0
    infer_count = 0
    frame_count = 0
    time_infer_total = 0
    time_all_start = time.time()
    # pipeline of inference
    for class_idx in range(0, len(input_directory)):   
        print("class: ", input_directory[class_idx])
        class_path = os.path.join(input_path, input_directory[class_idx])
        batch_id = 0
        input_numpy_array_b4 = []
        if os.path.isdir(class_path):
            video_path_list = os.listdir(class_path)
            video_path_list.sort(key = lambda x: x)
        for video_idx in range(0, len(video_path_list)):
            cap = cv2.VideoCapture(os.path.join(class_path, video_path_list[video_idx]))
            print("Read video path: ", os.path.join(class_path, video_path_list[video_idx]))
            frame_id = 0 
            input_numpy_array = []
            for i in range(0, max_video_length):
                ret, frame = cap.read()
                # ret, frame_ = cap.read()
                if ret == 0 or frame_id >= input_shape[2]:
                    break
                if i % step == 0:
                    
                    # if frame_id == 10:
                    #     cv2.imwrite("tmp.jpg", frame)
                    #     print("frame_id: ", i, frame)
                    #     return
                    
                    frame_count += 1
                    frame_id += 1
                    # frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                    tmp_ = center_crop(cv2.resize(frame, (171, 128)))
                    # tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                    # tmp = tmp_ - np.array([[[102.0, 98.0, 90.0]]])
                    tmp = tmp_ - np.array([[[104.0, 117.0, 123.0]]])
                    # tmp = tmp_ - np.array([[[123.0, 117.0, 104.0]]])
                    input_numpy_array.append(tmp)
            while len(input_numpy_array) < input_shape[2]:
                input_numpy_array.append(input_numpy_array[-1])
            if input_dtype == sail.BM_FLOAT32:
                input_numpy_array = np.array(input_numpy_array).astype(np.float32)
            elif input_dtype == sail.BM_INT8:
                input_numpy_array = np.array(input_numpy_array).astype(np.int8) 
            input_numpy_array = np.expand_dims(input_numpy_array, axis=0)
            input_numpy_array = np.transpose(input_numpy_array, (0, 4, 1, 2, 3))
            
            # print(input_numpy_array[0][0][0][0])
            # print(input_numpy_array[0][0][1][0])
            # print(input_numpy_array[0][0][2][0])
            if batch_size > 1:
                if batch_id == 0:
                    input_numpy_array_b4 = input_numpy_array
                else: 
                    input_numpy_array_b4 = np.concatenate((input_numpy_array_b4, input_numpy_array), axis=0)
                batch_id += 1
                if batch_id == batch_size:
                    input = sail.Tensor(engine.get_handle(), input_numpy_array_b4)
                    input_tensors = {input_name: input}
                    start_infer = time.time()
                    engine.process(graph_name, input_tensors, output_tensors)
                    end_infer = time.time()
                    infer_count += batch_size
                    
                    for output_ in output.asnumpy():
                        pred_idx = np.argmax(output_)
                        if pred_idx == class_idx:
                            correct_count += 1
                        total_count += 1
                        print(pred_idx, input_directory[pred_idx])
                    time_infer_total += end_infer - start_infer
                    batch_id = 0
                    input_numpy_array_b4 = []
            else:
                input = sail.Tensor(engine.get_handle(), input_numpy_array)
                input_tensors = {input_name: input}
                start_infer = time.time()
                engine.process(graph_name, input_tensors, output_tensors)
                end_infer = time.time()
                infer_count += batch_size
                time_infer_total += end_infer - start_infer
                pred_idx = np.argmax(output.asnumpy())
                if pred_idx == class_idx:
                    correct_count += 1
                total_count += 1
                print(pred_idx, input_directory[pred_idx])
        # if not finished
        if len(input_numpy_array_b4) != 0:
            resume_num = len(input_numpy_array_b4)
            # create fake data
            while len(input_numpy_array_b4) < batch_size:
                input_numpy_array_b4 = np.concatenate((input_numpy_array_b4, input_numpy_array), axis=0)
            input = sail.Tensor(engine.get_handle(), np.array(input_numpy_array_b4))
            input_tensors = {input_name: input}
            start_infer = time.time()
            engine.process(graph_name, input_tensors, output_tensors)
            end_infer = time.time()
            infer_count += batch_size
            for output_ in output.asnumpy()[0:resume_num]:
                pred_idx = np.argmax(output_)
                if pred_idx == class_idx:
                    correct_count += 1
                total_count += 1
                print(pred_idx, input_directory[pred_idx])
            time_infer_total += end_infer - start_infer
        print("acc now: ", correct_count / total_count)
    time_all_end = time.time()
    total_time = time_all_end - time_all_start
    print("total_time(ms): {:.2f}, frame_num: {}".format(total_time * 1000, frame_count))
    avg_infer_time = time_infer_total / infer_count
    print("avg_infer_time(ms): {:.2f}".format(batch_size * avg_infer_time * 1000))
    print("ACC: ",correct_count / total_count)
if __name__ == '__main__':
    """ A C3D example using bm-ffmpeg to decode and bmcv to preprocess.
    """
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--bmodel', default='../data/models/BM1684X/c3d_fp32_1b.bmodel', required=False)
    PARSER.add_argument('--input_path', default='../data/UCF_test_01', required=False)
    PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
    
    ARGS = PARSER.parse_args()
    if not (os.path.isdir(ARGS.input_path)):
        raise Exception('{} is not a valid input.'.format(ARGS.input_path))

    inference(ARGS.bmodel, ARGS.input_path, ARGS.tpu_id)
