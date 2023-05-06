#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# -*- coding: utf-8 -*- 
import os
import time
import cv2
import math
import numpy as np
import argparse
import json
import sophon.sail as sail
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0], [255, 0, 255], [255, 85, 255], \
          [255, 170, 255], [255, 255, 255], [170, 255, 255], [85, 255, 255], [0, 255, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170]]

class Pose(object):
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        logging.debug("load {} success!".format(args.bmodel))

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.stride = self.input_shape[2] / self.output_shape[2]
        self.point_num = int(self.output_shape[1] / 3) - 1
        
        self.pad_value = 128
        self.thre1 = 0.1
        self.thre2 = 0.05
        
        if self.point_num == 18:
            self.POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], \
                               [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2, 16], [5, 17]]
            self.mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], \
                           [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
        if self.point_num == 25:
            self.POSE_PAIRS = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], \
                               [10, 11], [8, 12], [12, 13], [13, 14], [1, 0], [0, 15], [15, 17], [0, 16], [16, 18], \
                               [2, 17], [5, 18], [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]]
            self.mapIdx = [[26,27], [40,41], [48,49], [42,43], [44,45], [50,51], [52,53], [32,33], [28,29], \
                           [30,31], [34,35], [36,37], [38,39], [56,57], [58,59], [62,63], [60,61], [64,65], \
                           [46,47], [54,55], [66,67], [68,69], [70,71], [72,73], [74,75], [76,77]]

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
    
    def preprocess(self, ori_img):
        h, w, _ = ori_img.shape
        h_scale = self.net_h / h
        w_scale = self.net_w / w
        scale = min(h_scale, w_scale)
        resize_img = cv2.resize(ori_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        top = 0 
        bottom = self.net_h - resize_img.shape[0]
        left = 0 
        right = self.net_w - resize_img.shape[1]
        pad_img = cv2.copyMakeBorder(resize_img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(self.pad_value,self.pad_value,self.pad_value))
        img = pad_img.astype('float32')
        img -= 128
        img /= 255
        img = np.transpose(img, (2, 0, 1))
        return img, [bottom, right]

    def predict(self, input_img):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)
        return list(outputs.values())[0]

    def postprocess(self, outputs, img_list, pad_list):
        results = []
        for i, output in enumerate(outputs):
            ori_img = img_list[i]
            pad = pad_list[i]
            
            output = np.transpose(output, (1, 2, 0))
            output = cv2.resize(output, (0, 0), fx=self.stride, fy=self.stride, interpolation=cv2.INTER_CUBIC)
            output = output[:self.net_h - pad[0], :self.net_w - pad[1], :]
            output = cv2.resize(output, (ori_img.shape[1], ori_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            all_peaks = []
            peak_counter = 0
            
            for part in range(self.point_num):
                map_ori = output[:, :, part]
                one_heatmap = gaussian_filter(map_ori, sigma=3)
                
                map_left = np.zeros(one_heatmap.shape)
                map_left[1:, :] = one_heatmap[:-1, :]
                map_right = np.zeros(one_heatmap.shape)
                map_right[:-1, :] = one_heatmap[1:, :]
                map_up = np.zeros(one_heatmap.shape)
                map_up[:, 1:] = one_heatmap[:, :-1]
                map_down = np.zeros(one_heatmap.shape)
                map_down[:, :-1] = one_heatmap[:, 1:]
                
                peaks_binary = np.logical_and.reduce(
                    (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > self.thre1))
                peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
                peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
                peak_id = range(peak_counter, peak_counter + len(peaks))
                peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

                all_peaks.append(peaks_with_score_and_id)
                peak_counter += len(peaks)
            
            connection_all = []
            special_k = []
            mid_num = 10
            
            for k in range(len(self.mapIdx)):
                score_mid = output[:, :, [x for x in self.mapIdx[k]]]
                candA = all_peaks[self.POSE_PAIRS[k][0]]
                candB = all_peaks[self.POSE_PAIRS[k][1]]
                nA = len(candA)
                nB = len(candB)
                indexA, indexB = np.array(self.POSE_PAIRS[k]) + 1
                if (nA != 0 and nB != 0):
                    connection_candidate = []
                    for i in range(nA):
                        for j in range(nB):
                            vec = np.subtract(candB[j][:2], candA[i][:2])
                            norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                            norm = max(0.001, norm)
                            vec = np.divide(vec, norm)

                            startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                                np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                            vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                            for I in range(len(startend))])
                            vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                            for I in range(len(startend))])

                            score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                            score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                                0.5 * ori_img.shape[0] / norm - 1, 0)
                            criterion1 = len(np.nonzero(score_midpts > self.thre2)[0]) > 0.8 * len(score_midpts)
                            criterion2 = score_with_dist_prior > 0
                            if criterion1 and criterion2:
                                connection_candidate.append(
                                    [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                    connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                    connection = np.zeros((0, 5))
                    for c in range(len(connection_candidate)):
                        i, j, s = connection_candidate[c][0:3]
                        if (i not in connection[:, 3] and j not in connection[:, 4]):
                            connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                            if (len(connection) >= min(nA, nB)):
                                break

                    connection_all.append(connection)
                else:
                    special_k.append(k)
                    connection_all.append([])

            # last number in each row is the total parts number of that person
            # the second last number in each row is the score of the overall configuration
            subset = -1 * np.ones((0, self.point_num + 2))
            candidate = np.array([item for sublist in all_peaks for item in sublist])

            for k in range(len(self.mapIdx)):
                if k not in special_k:
                    partAs = connection_all[k][:, 0]
                    partBs = connection_all[k][:, 1]
                    indexA, indexB = self.POSE_PAIRS[k]

                    for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                        found = 0
                        subset_idx = [-1, -1]
                        for j in range(len(subset)):  # 1:size(subset,1):
                            if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                                subset_idx[found] = j
                                found += 1

                        if found == 1:
                            j = subset_idx[0]
                            if subset[j][indexB] != partBs[i]:
                                subset[j][indexB] = partBs[i]
                                subset[j][-1] += 1
                                subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                        elif found == 2:  # if found 2 and disjoint, merge them
                            j1, j2 = subset_idx
                            membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                            if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                                subset[j1][:-2] += (subset[j2][:-2] + 1)
                                subset[j1][-2:] += subset[j2][-2:]
                                subset[j1][-2] += connection_all[k][i][2]
                                subset = np.delete(subset, j2, 0)
                            else:  # as like found == 1
                                subset[j1][indexB] = partBs[i]
                                subset[j1][-1] += 1
                                subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                        # if find no partA in the subset, create a new subset
                        elif not found and k < self.point_num - 1:
                            row = -1 * np.ones(self.point_num + 2)
                            row[indexA] = partAs[i]
                            row[indexB] = partBs[i]
                            row[-1] = 2
                            row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                            subset = np.vstack([subset, row])
            # delete some rows of subset which has few parts occur
            deleteIdx = []
            for i in range(len(subset)):
                if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                    deleteIdx.append(i)
            subset = np.delete(subset, deleteIdx, axis=0)

            # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
            # candidate: x, y, score, id
            results.append((candidate, subset))    

        return results

    def __call__(self, img_list):
        img_num = len(img_list)
        img_input_list = []
        pad_list = []
        for ori_img in img_list:
            start_time = time.time()
            img, pad = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            img_input_list.append(img)
            pad_list.append(pad)
        
        if img_num == self.batch_size:
            input_img = np.stack(img_input_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(img_input_list)
            
        start_time = time.time()
        outputs = self.predict(input_img)[:img_num]
        self.inference_time += time.time() - start_time
        
        start_time = time.time()
        results = self.postprocess(outputs, img_list, pad_list)
        self.postprocess_time += time.time() - start_time
        
        return results
    
    def draw_pose(self, ori_img, candidate, subset):
        canvas = ori_img.copy()
        for i in range(self.point_num):
            for n in range(len(subset)):
                index = int(subset[n][i])
                if index == -1:
                    continue
                x, y = candidate[index][0:2]
                cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
        for i in range(len(self.POSE_PAIRS)):
            for n in range(len(subset)):
                index = subset[n][np.array(self.POSE_PAIRS[i])]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        return canvas

def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir)
        
    # initialize net
    pose = Pose(args)
    batch_size = pose.batch_size
    
    decode_time = 0.0
    # test images
    if os.path.isdir(args.input):
        img_list = []
        filename_list = []
        results_list = []
        cn = 0
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg','.png','.jpeg','.bmp','.webp']:
                    continue
                cn += 1
                img_file = os.path.join(root, filename)
                logging.info("{}, img_file: {}".format(cn, img_file))
                
                # decode
                start_time = time.time()
                src_img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)
                if src_img is None:
                    logging.error("{} imdecode is None.".format(img_file))
                    continue
                if len(src_img.shape) != 3:
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                decode_time += time.time() - start_time
                
                img_list.append(src_img)
                filename_list.append(filename)
                if len(img_list) == batch_size:
                    # predict
                    results = pose(img_list)
                    
                    for i, filename in enumerate(filename_list):
                        candidate, subset = results[i]
                        logging.info("predict person nums: {}".format(subset.shape[0]))
                        # save image
                        res_img = pose.draw_pose(img_list[i], candidate, subset)
                        cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                        
                        # save result
                        res_dict = dict()
                        res_dict['image_name'] = filename
                        res_dict['keypoints'] = []
                        for n in range(len(subset)):
                            for m in range(pose.point_num):
                                index = int(subset[n][m])
                                if index == -1:
                                    x, y, score = 0, 0, 0.0
                                else:
                                    x, y, score = candidate[index][0:3]
                                res_dict['keypoints'].append(x)
                                res_dict['keypoints'].append(y)
                                res_dict['keypoints'].append(score)
                        results_list.append(res_dict)
                        
                    img_list.clear()
                    filename_list.clear()
                    
        if len(img_list):
            results = pose(img_list)
            for i, filename in enumerate(filename_list):
                candidate, subset = results[i]
                logging.info("predict person nums: {}".format(subset.shape[0]))
                res_img = pose.draw_pose(img_list[i], candidate, subset)
                cv2.imwrite(os.path.join(output_img_dir, filename), res_img)
                res_dict = dict()
                res_dict['image_name'] = filename
                res_dict['keypoints'] = []
                for n in range(len(subset)):
                    for m in range(pose.point_num):
                        index = int(subset[n][m])
                        if index == -1:
                            x, y, score = 0.0, 0.0, 0.0
                        else:
                            x, y, score = candidate[index][0:3]
                        res_dict['keypoints'].append(x)
                        res_dict['keypoints'].append(y)
                        res_dict['keypoints'].append(score)
                results_list.append(res_dict)    

        # save results
        if args.input[-1] == '/':
            args.input = args.input[:-1]
        json_name = os.path.split(args.bmodel)[-1] + "_" + os.path.split(args.input)[-1] + "_opencv" + "_python_result.json"
        with open(os.path.join(output_dir, json_name), 'w') as jf:
            # json.dump(results_list, jf)
            json.dump(results_list, jf, indent=4, ensure_ascii=False)
        logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))
        
    # test video
    else:
        cap = cv2.VideoCapture()
        if not cap.open(args.input):
            raise Exception("can not open the video")
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print(fps, size)
        save_video = os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '.avi')
        out = cv2.VideoWriter(save_video, fourcc, fps, size)
        cn = 0
        frame_list = []
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            # cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.split(args.input)[1])[0] + '_' + str(cn) + '.jpg'), frame)
            decode_time += time.time() - start_time
            if not ret or frame is None:
                break
            frame_list.append(frame)
            if len(frame_list) == batch_size:
                results = pose(frame_list)
                for i, frame in enumerate(frame_list):
                    cn +=1
                    candidate, subset = results[i]
                    logging.info("{}, person nums: {}".format(cn, subset.shape[0]))
                    res_frame = pose.draw_pose(frame, candidate, subset)
                    out.write(res_frame)
                frame_list.clear()
        if len(frame_list):
            results = pose(frame_list)
            for i, frame in enumerate(frame_list):
                cn +=1
                candidate, subset = results[i]
                logging.info("{}, person nums: {}".format(cn, subset.shape[0]))
                res_frame = pose.draw_pose(frame, candidate, subset)
                out.write(res_frame)
        cap.release()
        out.release()
        logging.info("result saved in {}".format(save_video))   

    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = pose.preprocess_time / cn
    inference_time = pose.inference_time / cn
    postprocess_time = pose.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
    # average_latency = decode_time + preprocess_time + inference_time + postprocess_time
    # qps = 1 / average_latency
    # logging.info("average latency time(ms): {:.2f}, QPS: {:2f}".format(average_latency * 1000, qps))
      
def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684/pose_coco_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
    print('all done.')
