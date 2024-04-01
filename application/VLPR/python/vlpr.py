#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#


import sophon.sail as sail
import numpy as np
import threading
import time
import os
import queue
from multiprocessing import Process
import argparse
import logging
from logging.handlers import RotatingFileHandler

from chars import CHARS,CHARS_DICT

class MultiDecoderThread(object):
    def __init__(self, draw_images:bool, stress_test:bool, tpu_id:int, video_list:list, resize_type:sail.sail_resize_type, max_que_size:int, loop_count:int,process_id:int):
        self.draw_images = draw_images
        self.stress_test = stress_test
        self.loop_count = loop_count
        self.video_list = video_list
        self.channel_list = {}
        self.tpu_id = tpu_id
        self.process_id = process_id

        self.resize_type = resize_type
        self.resize_type_lprnet = sail.sail_resize_type.BM_RESIZE_TPU_LINEAR

        self.multiDecoder = sail.MultiDecoder(16, tpu_id)
        self.multiDecoder.set_local_flag(True)

        self.post_que = queue.Queue(max_que_size)
        self.image_que = queue.Queue(max_que_size)
        self.max_que_size = max_que_size

        self.exit_flag = False
        self.flag_lock = threading.Lock()
        
        for video_name in video_list:
            channel_index = self.multiDecoder.add_channel(video_name,0)
            print("Process {}  Add Channel[{}]: {}".format(process_id,channel_index,video_name))
            self.channel_list[channel_index] = video_name

        # yolov5
        self.alpha_beta = (1.0/255,0),(1.0/255,0),(1.0/255,0)

        # lprnet 
        self.input_scale_lprnet = 1 
        self.alpha_beta_lprnet = (
            tuple(x * 1 * 0.0078125 for x in [1, -127.5]),
            tuple(x * 1 * 0.0078125 for x in [1, -127.5]),
            tuple(x * 1 * 0.0078125 for x in [1, -127.5])
        )

    def restart_multidecoder(self):
        for key in self.channel_list:
            self.multiDecoder.reconnect(int(key))
            print("reconnect:",int(key))

    def get_exit_flag(self):
        self.flag_lock.acquire()
        flag_temp = self.exit_flag
        self.flag_lock.release()
        return flag_temp

    def InitProcess(self, yolo_bmodel:str,lprnet_bmodel:str,dete_threshold:float,nms_threshold:float):
        """通过sail.EngineImagePreProcess, 初始化yolov5、lprnet的预处理和推理接口;
        通过sail.algo_yolov5_post_cpu_opt_async, 初始化yolov5后处理接口;
        初始化并开启所有处理线程

        Args:
            yolo_bmodel (str): yolov5 bmodel路径
            lprnet_bmodel (str): lprnet bmodel路径
            dete_threshold (float): yolov5 detect_threshold
            nms_threshold (float): yolov5 nms_threshold
        """
        self.handle = sail.Handle(self.tpu_id)
        self.bmcv = sail.Bmcv(self.handle)
        

        # yolov5
        self.engine_image_pre_process = sail.EngineImagePreProcess(yolo_bmodel, self.tpu_id, 0)
        self.engine_image_pre_process.InitImagePreProcess(self.resize_type, True, self.max_que_size - int(self.max_que_size/4)+1, self.max_que_size - int(self.max_que_size/4)+1) # queue_in_size, queue_out_size, avoid too much npu mem usage
        self.engine_image_pre_process.SetPaddingAtrr()
        self.engine_image_pre_process.SetConvertAtrr(self.alpha_beta)
        output_names = self.engine_image_pre_process.get_output_names()
        self.yolo_batch_size = self.engine_image_pre_process.get_output_shape(output_names[0])[0]
        output_shapes = [self.engine_image_pre_process.get_output_shape(i) for i in output_names]
        self.yolov5_post_async = sail.algo_yolov5_post_cpu_opt_async([output_shapes[0],output_shapes[1],output_shapes[2]],640,640,8)

        # lprnet
        self.lprnet_engine_image_pre_process = sail.EngineImagePreProcess(lprnet_bmodel, self.tpu_id, 0)
        self.lprnet_engine_image_pre_process.InitImagePreProcess(self.resize_type_lprnet, True, 32 - int(self.max_que_size/2)+1, 32 - int(self.max_que_size/2)+1) # queue_in_size, queue_out_size
        self.lprnet_engine_image_pre_process.SetConvertAtrr(self.alpha_beta_lprnet)
        self.lprnet_output_names = self.lprnet_engine_image_pre_process.get_output_names()[0]

        # start thread
        thread_preprocess = threading.Thread(target=self.decoder_and_pushdata, args=(self.channel_list, self.multiDecoder, self.engine_image_pre_process))
        thread_inference = threading.Thread(target=self.Inferences_thread, args=(self.post_que, self.image_que))
        thread_postprocess = threading.Thread(target=self.post_process, args=(self.post_que, dete_threshold, nms_threshold))
        thread_lprnet = threading.Thread(target=self.lprnet_pre_and_process,args=(self.image_que,))
        thread_drawresult = threading.Thread(target=self.lprnet_post_and_draw_result)
        
        
        thread_postprocess.start()
        thread_preprocess.start()
        thread_lprnet.start() 
        thread_inference.start() 
        thread_drawresult.start()
       
    
    def decoder_and_pushdata(self, channel_list:list, multi_decoder:sail.MultiDecoder, PreProcessAndInference:sail.EngineImagePreProcess):
        """使用multidecoder进行多路解码, 解码后的图片将被直接push进入sail.EngineImagePreProcess进行预处理和推理

        Args:
            channel_list (list): 传入进行解码的多路视频
            multi_decoder (sail.MultiDecoder): 
            PreProcessAndInference (sail.EngineImagePreProcess): 
        """
        image_index = 0

        while True:
            if self.get_exit_flag():
                break
            for key in channel_list:
                if self.get_exit_flag():
                    break
                bmimg = sail.BMImage()
                ret = multi_decoder.read(int(key),bmimg)
                if ret == 0:
                    image_index += 1
                    PreProcessAndInference.PushImage(int(key),image_index, bmimg)
                else:
                    time.sleep(0.01)

        print("decoder_and_pushdata thread exit!")

    def Inferences_thread(self, post_queue:queue.Queue, img_queue:queue.Queue):
        """使用sail.EngineImagePreProcess进行推理, 推理后的数据将被push入post_queue, 原图将被push入img_queue

        Args:
            post_queue (queue.Queue): 
                存放模型推理后的一个batch的数据, 每个数据为
                [output_tensor_map, 
                channel_list,
                imageidx_list, 
                width_list,       
                height_list, 
                padding_atrr]
            img_queue (queue.Queue): 
                存放输入模型的原始图片数据, 每个数据为
                {(channel,imageidx_list[index]):ost_images[index]} 
        """
        while True:
            if self.get_exit_flag():
                break
            start_time = time.time()
            output_tensor_map, ost_images, channel_list ,imageidx_list, padding_atrr = self.engine_image_pre_process.GetBatchData(True)

            width_list = []
            height_list= []
            for index, channel in enumerate(channel_list):
                width_list.append(ost_images[index].width())
                height_list.append(ost_images[index].height())

            while post_queue.full():
                time.sleep(0.01)
                if self.get_exit_flag():
                    break
                continue
            post_queue.put([output_tensor_map,
                            channel_list,
                            imageidx_list,
                            width_list, 
                            height_list, 
                            padding_atrr],False)
            
            for index, channel in enumerate(channel_list):
                while img_queue.full():
                    time.sleep(0.01)
                    if self.get_exit_flag():
                        break
                    continue 
                img_queue.put({(channel,imageidx_list[index]):ost_images[index]}) 

                logging.debug("put ost img to queue, cid is {}, frameid is{}".format(channel,imageidx_list[index]))

            end_time = time.time()
            logging.info("Engine_image_pre_process GetBatchData time use: {:.2f} ms".format((end_time-start_time)*1000))
        
        print("Inferences_thread thread exit!")

    def post_process(self, post_quque:queue.Queue, dete_threshold:float, nms_threshold:float):
        """通过post_quque得到模型推理后的数据, 并将数据传入后处理

        Args:
            post_quque (queue.Queue): _description_
            dete_threshold (float): _description_
            nms_threshold (float): _description_
        """
        while (True):
            if self.get_exit_flag():
                break
            # if post_quque.empty():
            #     time.sleep(0.01)
            #     continue
            output_tensor_map, channels ,imageidxs, ost_ws, ost_hs, padding_atrrs = post_quque.get(True)

            dete_thresholds = np.ones(len(channels),dtype=np.float32)
            nms_thresholds = np.ones(len(channels),dtype=np.float32)
            dete_thresholds = dete_threshold*dete_thresholds
            nms_thresholds = nms_threshold*nms_thresholds
            while True:
                if self.get_exit_flag():
                    break
                ret = self.yolov5_post_async.push_data(channels, imageidxs, output_tensor_map, dete_thresholds, nms_thresholds, ost_ws, ost_hs, padding_atrrs)

                if ret == 0:
                    break
                else:
                    print("push_data failed, ret: {}".format(ret))
                    time.sleep(0.01)
                    # break
        print("post_process thread exit!")
    
    def lprnet_pre_and_process(self, img_queue:queue.Queue):
        """从self.yolov5_post_async得到yolov5后处理的一组数据输出, 并将其在原图上crop出小图, 
        送入self.lprnet_engine_image_pre_process进行推理;
        
        Args:
            img_queue (queue.Queue): 存放原图的队列
            
        """

        yolo_res_list = []
        draw_flag = False
        while (True):
            if self.get_exit_flag():
                break
            if img_queue.empty():
                time.sleep(0.01)
                continue

            ocv_image = img_queue.get(True) 
            objs, channel, image_idx = self.yolov5_post_async.get_result_npy() 

            # print("lprnet_pre_and_process: yolo post id and ocv id is ",(channel,image_idx),ocv_image.keys()) 

            if (channel,image_idx) == list(ocv_image.keys())[0]:
                img = list(ocv_image.values())[0]
                for obj in objs: # 一张图上多个结果
                    x1, y1, x2, y2, category_id, score = obj
                    
                    logging.debug("Lprnet_pre_and_process:Process {},channel_idx is {} image_idx is {},len(objs) is{}".format(self.process_id,channel, image_idx, len(objs)))
                    logging.debug("Lprnet_pre_and_process:Process %d,YOLO postprocess DONE! objs:tuple[left, top, right, bottom, class_id, score] :%s",self.process_id,obj)

                    if((x2-x1) <=16 or (y2 - y1) <= 16):
                        pass
                    else:
                        croped = self.bmcv.crop(img,int(x1),int(y1),int(x2-x1),int(y2-y1))
                        self.lprnet_engine_image_pre_process.PushImage(channel, image_idx, croped)

                    # draw images
                    if self.draw_images:
                        if x1 in yolo_res_list:
                            draw_flag = False
                            pass
                        else:
                            yolo_res_list.append(x1)
                            draw_flag = True
                            self.bmcv.rectangle(img, obj[0], obj[1], obj[2]-obj[0], obj[3]-obj[1],(0,0,255),2)
                        if len(yolo_res_list) >= 5:
                            yolo_res_list.clear()
                if draw_flag:
                    image = sail.BMImage(self.handle,img.height(),img.width(),sail.Format.FORMAT_YUV420P,sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
                    self.bmcv.convert_format(img,image)
                    self.bmcv.imwrite("c{}_f{}__P{}.jpg".format(channel,image_idx,self.process_id),image)
            else:
                logging.error("lprnet_pre_and_process: yolo post result idx, is not equal to origin images idx:")
                logging.error((channel,image_idx) ,list(ocv_image.keys())[0])

        print("Lprnet_pre_and_process thread exit!")

    def lprnet_post_and_draw_result(self):
        """通过self.lprnet_engine_image_pre_process获取lprnet后处理的数据;
        
        """
        file = open('lp_result.txt', 'w')
        template_infos = {}
        template_in_threshold = 4
        template_out_thresh = 4

        start_time = time.time()
        while (True):
            rm_list = []
            # 1 get lprnet process res
            if self.get_exit_flag():
                break
            output, _, channel_list, image_idx_list,_ = self.lprnet_engine_image_pre_process.GetBatchData_Npy() 
            logging.debug("Lprnet_post:Process {},channel_idx is {} image_idx is {}".format(self.process_id,channel_list, image_idx_list))

            output_array = output[self.lprnet_output_names][:4]

            res = list() # lprnet的batch=4时res长度为4
            for temp in np.argmax(output_array, axis=1):
                no_repeat_blank_label = list()
                pre_c = temp[0]
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(CHARS_DICT[pre_c])
                for c in temp:
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(CHARS_DICT[c])
                    pre_c = c
                res.append("".join(no_repeat_blank_label))
            print(res)
            logging.info('Process {}, LPRNET POSTPROCESS DONE,res{}'.format(self.process_id, res))

            # remove repeat lp 
            for i  in range(len(res)):
                lp_name = res[i]
                cid = channel_list[i] 
                fid = image_idx_list[i]
                if lp_name in template_infos.keys():
                    template_infos[lp_name]["in"]+=1
                    if template_infos[lp_name]["in"]==template_in_threshold:
                        # up_list.append(lp_res)      
                        file.write(f"process{self.process_id}:cid {cid},fid {fid},recongized license plates {lp_name} \n")
                else:
                    template_infos[lp_name]={}
                    template_infos[lp_name]["in"]=1
                    if template_infos[lp_name]["in"]==template_in_threshold:
                        # up_list.append(lp_res)
                        file.write(f"process{self.process_id}:cid {cid},fid {fid},recongized license plates {lp_name} \n")
                        
                for key in template_infos.keys():
                    if key != lp_name:
                        if "out" in template_infos[key].keys():
                            template_infos[key]["out"]+=1
                            if template_infos[key]["out"]>=template_out_thresh:
                                rm_list.append(key)       
                        else:
                            template_infos[key]["out"]=1
                if len(rm_list) :
                    for key in rm_list:
                        # print(key,rm_list,template_infos)
                        if key in template_infos:
                            del template_infos[key]

            if self.loop_count <=  image_idx_list[-1]: 
                logging.info("LOOPS DONE")
                end_time = time.time()
                time_use = (end_time-start_time)*1000
                avg_time = time_use/image_idx_list[-1]

                print("Process {}:Total images: {} ms".format(self.process_id, self.loop_count))
                print("Total time use: {} ms".format(time_use))
                print("Avg time use: {} ms".format(avg_time))
                print("Process {}: {} FPS".format(self.process_id, 1000/avg_time))
                print("Result thread exit!")

                logging.info("Process {}:Loops{},Total time use: {} ms, avg_time{}, this process is{} FPS".format(self.process_id, self.loop_count,time_use,avg_time,1000/avg_time))
                print("Process {}:Loops{},Total time use: {} ms, avg_time{}, this process is {} FPS".format(self.process_id, self.loop_count,time_use,avg_time,1000/avg_time))

                if not self.stress_test:
                    self.flag_lock.acquire()
                    self.exit_flag = True
                    self.flag_lock.release()
                    os._exit(1)
                    break
                elif self.stress_test:
                    self.loop_count += self.loop_count
                    self.restart_multidecoder()
                    pass

            

        

def process_demo(draw_images,stress_test,tpu_id, max_que_size, video_name_list, yolo_bmodel,lprnet_bmodel, loop_count, process_id,dete_threshold,nms_threshold):
    process =  MultiDecoderThread(draw_images,stress_test,tpu_id, video_name_list, sail.sail_resize_type.BM_PADDING_TPU_LINEAR, max_que_size, loop_count,process_id)
    process.InitProcess(yolo_bmodel,lprnet_bmodel,dete_threshold,nms_threshold)


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--max_que_size', type=int, default=4, help='multidecode queue')
    parser.add_argument('--video_nums', type=int, default=16, help='procress nums of input')
    parser.add_argument('--batch_size', type=int, default=4, help='video_nums/batch_size is procress nums of process and postprocess')
    parser.add_argument('--loops', type=int, default=1000, help='process loops for one process')
    parser.add_argument('--input', type=str, default='../datasets/licenseplate_640516-h264.mp4', help='path of input, must be video path') 
    parser.add_argument('--yolo_bmodel', type=str, default='../models/yolov5s-licensePlate/BM1684X/yolov5s_v6.1_license_3output_int8_4b.bmodel', help='path of bmodel')
    parser.add_argument('--lprnet_bmodel', type=str, default='../models/lprnet/BM1684X/lprnet_int8_4b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='tpu id')
    parser.add_argument('--draw_images', type=bool, default=False, help='draw images or not')
    parser.add_argument('--stress_test', type=bool, default=False, help='stress test or not')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = argsparser()

    # 配置log
    log_name = f'168X_yolo_process_and_video_thread_is_{args.video_nums}.log'
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    rotating_handler = RotatingFileHandler(log_name, maxBytes=1024*1024, backupCount=3)
    rotating_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(rotating_handler)
    logging.getLogger().setLevel(logging.INFO)
    

    dete_threshold,nms_threshold = 0.65,0.65
    max_que_size = args.max_que_size  # 队列缓存的大小
    loop_count = args.loops # 每个进程处理图片的数量，处理完毕之后会退出
    
    process_nums = int(args.video_nums/args.batch_size)
    input_videos = [args.input for _ in range(int(args.video_nums/process_nums))] # 初始化多路本地视频流


    decode_yolo_processes = [Process(target=process_demo,args=(args.draw_images,args.stress_test,args.dev_id, max_que_size, input_videos, args.yolo_bmodel,args.lprnet_bmodel, loop_count, i,dete_threshold,nms_threshold)) for i in range(process_nums) ]
    for i in decode_yolo_processes:
        i.start()
        logging.debug('start decode and yolo process')
    start_time = time.time()

    logging.info(start_time)
    for i in decode_yolo_processes:
        i.join()
        logging.debug('DONE decode and yolo process')

    total_time = time.time() - start_time
    if args.stress_test:
        pass
    else:
        print('video nums{}, process is {},total time is {},loops for one process is {},total fps is {}'.format(args.video_nums,process_nums,total_time,loop_count,(loop_count*process_nums)/total_time))
        logging.info('video nums{}, process is {},total time is {},loops for one process is {},total fps is {}'.format(args.video_nums,process_nums,total_time,loop_count,(loop_count*process_nums)/total_time))

