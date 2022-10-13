import os
import sys
os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

import os
import shutil
import numpy as np
import cv2
import argparse
import sophon.sail as sail
from yolov5_utils.preprocess_bmcv import PreProcess
from yolov5_utils.postprocess_numpy import PostProcess
from yolov5_utils.sophon_inference import SophonInference
from yolov5_utils.utils import draw_bmcv, draw_numpy, is_img


class YOLOv5:
    def __init__(self, model_path, device_id, conf_thresh=0.5, nms_thresh=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError('{} is not existed.'.format(model_path))

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.net = SophonInference(
            model_path=model_path,
            device_id=device_id,
            input_mode=1, # use bmcv
        )
        
        self.device_id = device_id

        self.bmcv = self.net.bmcv
        self.handle = self.net.handle
        self.input_scale = list(self.net.input_scales.values())[0]
        self.img_dtype = list(self.net.img_dtypes.values())[0]

        self.batch_size = self.net.inputs_shapes[0][0]
        self.net_c = self.net.inputs_shapes[0][1]
        self.net_h = self.net.inputs_shapes[0][2]
        self.net_w = self.net.inputs_shapes[0][3]
        self.preprocess = PreProcess(
            self.net_w,
            self.net_h,
            self.batch_size,
            self.img_dtype,
            self.input_scale,
        )

        self.agnostic = False
        self.multi_label = True
        self.max_det = 1000
        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )

        self.resort = False
        self.output_order_node = ['147', '148', '149'] # if use resort, please set output_order_node manually

        print('{} is loaded.'.format(model_path))


    def predict(self, tensor):
        """
        ensure output order: loc_data, conf_preds, mask_data, proto_data
        Args:
            tensor:

        Returns:

        """
        # feed: [input0]

        out_dict = self.net.infer_bmimage(tensor)
        if not self.resort:
            self.output_order_node = self.net.output_names

        # resort
        out_keys = list(out_dict.keys())
        ord = []
        for n in self.output_order_node:
            for i, k in enumerate(out_keys):
                if n in k:
                    ord.append(i)
                    break
        out = [out_dict[out_keys[i]] for i in ord]
        return out
    
    def do_once_proc(self, file_path):
        
        batch_size = self.batch_size
        input_path = file_path

        if not os.path.exists(input_path):
            raise FileNotFoundError('{} is not existed.'.format(input_path))
        
        # imgage directory
        input_list = []
        assert is_img(input_path), "not correct img path: {}".format(input_path)
        input_list.append(input_path)
        # imgage list saved in file

        inp_batch = []
        images = []
        ino = 0
        image = sail.BMImage()        
        ret = decode_image_bmcv(input_list[ino], self.handle, image, self.device_id)
        if not ret:
            # decode failed.
            print('skip: decode failed: {}'.format(input_list[ino]))
            return None
        
        images.append(image)
        inp_batch.append(input_list[ino])

        if batch_size == 1:
            single_image = images[0]
            org_h, org_w = single_image.height(), single_image.width()
            # end-to-end inference
            preprocessed_img, ratio, txy = self.preprocess(
                single_image,
                self.handle,
                self.bmcv,
            )

            out_infer = self.predict([preprocessed_img])

            det_batch = self.postprocess.infer_batch(
                out_infer,
                [(org_w, org_h)],
                [ratio],
                [txy],
            )

            det = det_batch[0]
        else:
            print("eval just support 1 batch, actual is {}".format(batch_size))
            det = None
        return det


def decode_image_bmcv(image_path, process_handle, img, dev_id):
    # img = sail.BMImage()
    # img = sail.BMImageArray4D()
    decoder = sail.Decoder(image_path, True, dev_id)
    if isinstance(img, sail.BMImage):
        ret = decoder.read(process_handle, img)
    else:
        ret = decoder.read_(process_handle, img)
    if ret != 0:
        return False
    return True


def main(opt):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    else:
        shutil.rmtree(opt.output_dir)
        os.makedirs(opt.output_dir)

    yolov5 = YOLOv5(
        model_path=opt.model,
        device_id=opt.dev_id,
        conf_thresh=opt.conf_thresh,
        nms_thresh=opt.nms_thresh,
    )

    batch_size = yolov5.batch_size
    input_path = opt.input_path

    if not os.path.exists(input_path):
        raise FileNotFoundError('{} is not existed.'.format(input_path))

    if opt.is_video:
        if batch_size != 1:
            raise ValueError(
                'bmodel batch size must be 1 in video inference, but got {}'.format(
                    batch_size)
            )
        # decode
        decoder = sail.Decoder(input_path, True, opt.dev_id)
        if decoder.is_opened():
            print("create decoder success")
            frame = sail.BMImage()
            id = 0
            while True:
                ret = decoder.read(yolov5.handle, frame)
                if ret:
                    print("stream end or decoder error")
                    break

                org_h, org_w = frame.height(), frame.width()

                preprocessed_img, ratio, txy = yolov5.preprocess(
                    frame,
                    yolov5.handle,
                    yolov5.bmcv,
                )

                out_infer = yolov5.predict([preprocessed_img])

                det_batch = yolov5.postprocess.infer_batch(
                    out_infer,
                    [(org_w, org_h)],
                    [ratio],
                    [txy],
                )
                det = det_batch[0]

                image_rgb_planar = sail.BMImage(yolov5.handle, frame.height(), frame.width(),
                                                sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                yolov5.net.bmcv.convert_format(frame, image_rgb_planar)
                draw_bmcv(yolov5.bmcv, image_rgb_planar, det[:, :4],
                          masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
                
                save_basename, _ = os.path.splitext(os.path.basename(opt.model))
                save_basename = save_basename + '_bmcv_python_result_{}.jpg'.format(id)
                save_name = os.path.join(opt.output_dir, save_basename)
                yolov5.bmcv.imwrite(save_name, image_rgb_planar)

                # convert BMImage to numpy to draw
                # image_bgr_planar = sail.BMImage(yolov5.handle, frame.height(), frame.width(),
                #                                 sail.Format.FORMAT_BGR_PLANAR, frame.dtype())
                # yolov5.bmcv.convert_format(frame, image_bgr_planar)
                # image_tensor = yolov5.bmcv.bm_image_to_tensor(image_bgr_planar)
                # image_chw_numpy = image_tensor.asnumpy()[0]
                # image_numpy = np.transpose(image_chw_numpy, [1, 2, 0]).copy()
                #
                # vis_image = draw_numpy(image_numpy, det[:,:4],
                #                        masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
                #
                # save_basename = 'res_bmcv_{}'.format(id)
                # save_name = os.path.join(opt.output_dir, save_basename.replace('.jpg', ''))
                # cv2.imencode('.jpg', vis_image)[1].tofile('{}.jpg'.format(save_name))

                id += 1

        else:
            print("failed to create decoder")

    else:
        # imgage directory
        input_list = []
        if os.path.isdir(input_path):
            for img_name in os.listdir(input_path):
                if is_img(img_name):
                    input_list.append(os.path.join(input_path, img_name))
                    # imgage file
        elif is_img(input_path):
            input_list.append(input_path)
        # imgage list saved in file
        else:
            with open(input_path, 'r', encoding='utf-8') as fin:
                for line in fin.readlines():
                    line_head = line.strip("\n").split(' ')[0]
                    if is_img(line_head):
                        input_list.append(line_head)

        img_num = len(input_list)

        suppoort_batch_size = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        if batch_size not in suppoort_batch_size:
            raise NotImplementedError(
                'model batch size must be {}, but got {}.'.format(suppoort_batch_size, batch_size))

        inp_batch = []
        images = []

        for ino in range(img_num):
            image = sail.BMImage()
            ret = decode_image_bmcv(input_list[ino], yolov5.handle, image, opt.dev_id)
            if not ret:
                # decode failed.
                print('skip: decode failed: {}'.format(input_list[ino]))
                continue
            images.append(image)
            inp_batch.append(input_list[ino])

            if len(images) != batch_size and ino != (img_num - 1):
                continue

            if batch_size == 1:
                single_image = images[0]
                org_h, org_w = single_image.height(), single_image.width()
                # end-to-end inference
                preprocessed_img, ratio, txy = yolov5.preprocess(
                    single_image,
                    yolov5.handle,
                    yolov5.bmcv,
                )

                out_infer = yolov5.predict([preprocessed_img])

                det_batch = yolov5.postprocess.infer_batch(
                    out_infer,
                    [(org_w, org_h)],
                    [ratio],
                    [txy],
                )

                det = det_batch[0]

                image_rgb_planar = yolov5.net.bmcv.convert_format(single_image)
                draw_bmcv(yolov5.bmcv, image_rgb_planar, det[:,:4],
                          masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
                save_basename, _ = os.path.splitext(os.path.basename(opt.model)) 
                input_name, _ = os.path.splitext(os.path.basename(inp_batch[0])) 
                save_basename = save_basename + '_bmcv_python_result_{}.jpg'.format(input_name)
                save_name = os.path.join(opt.output_dir, save_basename)
                yolov5.bmcv.imwrite(save_name, image_rgb_planar)

                # convert BMImage to numpy to draw
                # image_bgr_planar = sail.BMImage(yolov5.handle, single_image.height(), single_image.width(),
                #                                 sail.Format.FORMAT_BGR_PLANAR, single_image.dtype())
                # yolov5.bmcv.convert_format(single_image, image_bgr_planar)
                # image_tensor = yolov5.bmcv.bm_image_to_tensor(image_bgr_planar)
                # image_chw_numpy = image_tensor.asnumpy()[0]
                # image_numpy = np.transpose(image_chw_numpy, [1, 2, 0]).copy()
                #
                # vis_image = draw_numpy(image_numpy, det[:,:4],
                #                        masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
                # save_basename = 'res_bmcv_{}'.format(os.path.basename(inp_batch[0]))
                # save_name = os.path.join(opt.output_dir, save_basename.replace('.jpg', ''))
                # cv2.imencode('.jpg', vis_image)[1].tofile('{}.jpg'.format(save_name))
                # print('use cv imwrite')

            else:
                # padding params
                cur_bs = len(images)
                padding_bs = batch_size - cur_bs
                # adjustment for BMImageArray
                bm_array = eval('sail.BMImageArray{}D'.format(batch_size))

                org_size_list = []
                for i in range(len(inp_batch)):
                    org_h, org_w = images[i].height(), images[i].width()
                    org_size_list.append((org_w, org_h))

                resized_imgs = bm_array(yolov5.handle,
                                        yolov5.net_h,
                                        yolov5.net_w,
                                        sail.FORMAT_RGB_PLANAR,
                                        sail.DATA_TYPE_EXT_1N_BYTE
                                        )
                # batch end-to-end inference
                resized_img_list, ratio_list, txy_list = yolov5.preprocess.resize_batch(
                    images,
                    yolov5.handle,
                    yolov5.bmcv,
                )

                for i in range(len(inp_batch)):
                    resized_imgs.copy_from(i, resized_img_list[i])

                # padding is not necessary for bmcv in preprcessing
                # for i in range(cur_bs, batch_size):
                #     resized_imgs.copy_from(i, resized_img_list[0])

                preprocessed_imgs = yolov5.preprocess.norm_batch(
                    resized_imgs,
                    yolov5.handle,
                    yolov5.bmcv,
                )

                out_infer = yolov5.predict([preprocessed_imgs])

                # cancel padding data
                if padding_bs != 0:
                    out_infer = [e_data[:cur_bs] for e_data in out_infer]

                det_batch = yolov5.postprocess.infer_batch(
                    out_infer,
                    org_size_list,
                    ratio_list,
                    txy_list,
                )

                for i, (e_img, det) in enumerate(zip(images,
                                                     det_batch,
                                                     )):
                    image_rgb_planar = sail.BMImage(yolov5.handle, e_img.height(), e_img.width(),
                                                    sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
                    yolov5.net.bmcv.convert_format(e_img, image_rgb_planar)
                    draw_bmcv(yolov5.bmcv, image_rgb_planar, det[:,:4],
                              masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
                    save_basename, _ = os.path.splitext(os.path.basename(opt.model)) 
                    input_name, _ = os.path.splitext(os.path.basename(inp_batch[i])) 
                    save_basename = save_basename + '_bmcv_python_result_{}.jpg'.format(input_name)
                    save_name = os.path.join(opt.output_dir, save_basename)
                    yolov5.bmcv.imwrite(save_name, image_rgb_planar)

                    # convert BMImage to numpy to draw
                    # image_bgr_planar = sail.BMImage(yolov5.handle, e_img.height(), e_img.width(),
                    #                                 sail.Format.FORMAT_BGR_PLANAR, e_img.dtype())
                    # yolov5.bmcv.convert_format(e_img, image_bgr_planar)
                    # image_tensor = yolov5.bmcv.bm_image_to_tensor(image_bgr_planar)
                    # image_chw_numpy = image_tensor.asnumpy()[0]
                    # image_numpy = np.transpose(image_chw_numpy, [1, 2, 0]).copy()
                    #
                    # vis_image = draw_numpy(image_numpy, det[:,:4],
                    #                        masks=None, classes_ids=det[:, -1], conf_scores=det[:, -2])
                    # save_basename = 'res_bmcv_{}'.format(os.path.basename(inp_batch[i]))
                    # save_name = os.path.join(opt.output_dir, save_basename.replace('.jpg', ''))
                    # cv2.imencode('.jpg', vis_image)[1].tofile('{}.jpg'.format(save_name))

            images.clear()
            inp_batch.clear()

        print('the results is saved: {}'.format(os.path.abspath(opt.output_dir)))

def parse_opt():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--model', type=str, default="../data/models/BM1684X/yolov5s_640_coco_v6.1_3output_fp32_1b.bmodel", help='bmodel path')
    parser.add_argument('--dev_id', type=int, default=0, help='device id')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--is_video', default=0, type=int, help="input is video?")
    parser.add_argument('--input_path', type=str, default="../data/images/zidane.jpg", help='input image path')
    parser.add_argument('--output_dir', type=str, default="results", help='output image directory')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    print('all done.')








