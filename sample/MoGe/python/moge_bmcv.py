#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import time
import json
import struct
import zlib
import argparse
import numpy as np
import sophon.sail as sail
import logging

from postprocess_numpy import end_to_end, colorize_depth, colorize_normal

logging.basicConfig(level=logging.INFO)


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def save_rgb_png(path, rgb):
    """Minimal RGB8 PNG writer (NumPy/zlib only, no cv2/PIL/scipy on the target board)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[0], rgb.shape[1]
    raw = np.empty((h, w * 3 + 1), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = rgb.reshape(h, w * 3)
    raw = raw.tobytes()

    def chunk(typ, payload):
        out = struct.pack('>I', len(payload)) + typ + payload
        out += struct.pack('>I', zlib.crc32(typ + payload) & 0xffffffff)
        return out

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)  # 8-bit, color type 2 = RGB
    png = sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
    with open(path, 'wb') as f:
        f.write(png)


class MoGe:
    def __init__(self, args):
        # load bmodel (SYSIO: numpy in / numpy out, all tensors are fp32)
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.debug("load {} success!".format(args.bmodel))
        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = _s(self.net.get_graph_names()[0])

        # get input
        self.input_name = _s(self.net.get_input_names(self.graph_name)[0])
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        # map bmodel outputs (order/names are not guaranteed) to canonical keys
        output_names = [_s(o) for o in self.net.get_output_names(self.graph_name)]
        self._out_key = {}
        for name in output_names:
            low = name.lower()
            if 'metric_scale' in low:
                self._out_key['metric_scale'] = name
            elif 'mask' in low:
                self._out_key['mask'] = name
            elif 'normal' in low:
                self._out_key['normal'] = name
            elif 'points' in low:
                self._out_key['points'] = name
        for k in ('points', 'normal', 'mask', 'metric_scale'):
            if k not in self._out_key:
                raise ValueError('could not locate output "{}" among {}.'.format(k, output_names))

        # MoGe is exported as a static single-image graph (1, 3, 518, 518)
        if self.input_shape[0] != 1 or self.input_shape[1] != 3:
            raise ValueError('MoGe bmodel must have input shape (1, 3, H, W), got {}.'.format(self.input_shape))
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess_bmcv(self, input_bmimg):
        """Decode-independent part: any source -> RGB float32 [0, 1] of shape (1, 3, H, W)."""
        # arbitrary source (BGR packed from decoder) -> RGB planar uint8
        rgb_planar = sail.BMImage(self.handle, input_bmimg.width(), input_bmimg.height(),
                                  sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar)
        # direct resize to the fixed square input (MoGe aspect-ratio = 1, no letter-box)
        resized = self.bmcv.resize(rgb_planar, self.net_w, self.net_h,
                                   sail.bmcv_resize_algorithm.BMCV_INTER_LINEAR)
        # resize output may be non-contiguous; identity-scale convert_to makes a contiguous fp32 copy.
        # the /255 normalization is done in NumPy so it is exact (bmcv fixed-point scale would be /256).
        fp_image = sail.BMImage(self.handle, self.net_w, self.net_h,
                                sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized, fp_image, ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)))
        input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
        self.bmcv.bm_image_to_tensor(fp_image, input_tensor)
        return np.ascontiguousarray(input_tensor.asnumpy(), dtype=np.float32) / 255.0

    def __call__(self, bmimg):
        start_time = time.time()
        x = self.preprocess_bmcv(bmimg)
        self.preprocess_time += time.time() - start_time

        start_time = time.time()
        raw = self.net.process(self.graph_name, {self.input_name: x})
        self.inference_time += time.time() - start_time

        start_time = time.time()
        outputs = {k: np.asarray(raw[_s(self._out_key[k])], dtype=np.float32) for k in self._out_key}
        points = outputs['points'].squeeze(0)          # (H, W, 3)
        normal = outputs['normal'].squeeze(0)          # (H, W, 3)
        mask = outputs['mask'].squeeze(0)              # (H, W)
        metric_scale = outputs['metric_scale'].ravel()  # (1,)

        result = end_to_end(points, mask, metric_scale, force_projection=True)
        result['normal'] = normal
        self.postprocess_time += time.time() - start_time
        return result


def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))

    output_dir = "./results"
    output_img_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_img_dir, exist_ok=True)

    moge = MoGe(args)
    moge.init()

    file_list = []
    if os.path.isdir(args.input):
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.webp']:
                    file_list.append(os.path.join(root, filename))
    else:
        file_list.append(args.input)

    handle = sail.Handle(args.dev_id)
    cn = 0
    decode_time = 0.0
    results_list = []
    for img_file in file_list:
        cn += 1
        logging.info("{}, img_file: {}".format(cn, img_file))

        decoder = sail.Decoder(img_file, True, args.dev_id)
        bmimg = sail.BMImage()
        start_time = time.time()
        ret = decoder.read(handle, bmimg)
        decode_time += time.time() - start_time
        if ret != 0:
            logging.error("{} decode failure.".format(img_file))
            continue

        result = moge(bmimg)

        fname = os.path.splitext(os.path.split(img_file)[1])[0]
        depth = result['depth']
        mask = result['mask']

        np.save(os.path.join(output_dir, '{}_depth.npy'.format(fname)), depth.astype(np.float32))
        np.save(os.path.join(output_dir, '{}_points.npy'.format(fname)), result['points'].astype(np.float32))
        np.save(os.path.join(output_dir, '{}_normal.npy'.format(fname)),
                result['normal'].astype(np.float32).transpose(2, 0, 1))
        save_rgb_png(os.path.join(output_img_dir, '{}_depth.png'.format(fname)), colorize_depth(depth, mask))
        save_rgb_png(os.path.join(output_img_dir, '{}_normal.png'.format(fname)),
                     colorize_normal(result['normal'], mask))

        res_dict = {
            'image_name': os.path.split(img_file)[1],
            'intrinsics': result['intrinsics'].flatten().tolist(),
            'fov_x_deg': result['fov_x_deg'],
            'fov_y_deg': result['fov_y_deg'],
            'focal': result['focal'],
            'shift': result['shift'],
            'depth_min': float(depth[mask].min()) if mask.any() else 0.0,
            'depth_max': float(depth[mask].max()) if mask.any() else 0.0,
        }
        results_list.append(res_dict)
        logging.info("  fov_x={:.3f}deg fov_y={:.3f}deg depth=[{:.3f}, {:.3f}]m".format(
            result['fov_x_deg'], result['fov_y_deg'], res_dict['depth_min'], res_dict['depth_max']))

    json_name = os.path.splitext(os.path.split(args.bmodel)[1])[0] + "_python_result.json"
    with open(os.path.join(output_dir, json_name), 'w') as jf:
        json.dump(results_list, jf, indent=4, ensure_ascii=False)
    logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

    logging.info("------------------ Predict Time Info ----------------------")
    decode_time = decode_time / cn
    preprocess_time = moge.preprocess_time / cn
    inference_time = moge.inference_time / cn
    postprocess_time = moge.postprocess_time / cn
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input image or directory')
    parser.add_argument('--bmodel', type=str, default='../models/BM1688/moge-2-vitl-normal_fp16_1b.bmodel',
                        help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')