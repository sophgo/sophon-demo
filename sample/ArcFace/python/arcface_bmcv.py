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
import argparse
import numpy as np
import sophon.sail as sail
import logging

logging.basicConfig(level=logging.INFO)


class ArcFace:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSO)
        logging.debug("load {} success!".format(args.bmodel))

        self.handle = sail.Handle(args.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        self.graph_name = self.net.get_graph_names()[0]

        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_shapes = {self.input_name: self.input_shape}

        self.output_names = self.net.get_output_names(self.graph_name)
        if len(self.output_names) != 1:
            raise ValueError('only support 1 output, but got {} outputs'.format(len(self.output_names)))

        self.output_name = self.output_names[0]
        self.output_shape = self.net.get_output_shape(self.graph_name, self.output_name)
        self.output_dtype = self.net.get_output_dtype(self.graph_name, self.output_name)
        self.output_tensor = sail.Tensor(self.handle, self.output_shape, self.output_dtype, True, True)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        # Preprocess params: mean/scale is handled by bmodel internally
        # bmcv convert_to just does format conversion, not normalization
        self.ab = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess_bmcv(self, input_bmimg):
        """Preprocess: Resize to 112x112, convert to RGB planar float32"""
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                       sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)

        # Resize to model input size (directly, no keep aspect ratio for ArcFace)
        resized_img_rgb = self.bmcv.resize(rgb_planar_img, self.net_w, self.net_h)

        # Convert to model input dtype (float32)
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w,
                                          sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg,
                             ((self.ab[0], self.ab[1]),
                              (self.ab[2], self.ab[3]),
                              (self.ab[4], self.ab[5])))
        return preprocessed_bmimg

    def predict(self, input_tensor):
        """Run bmodel inference"""
        input_tensors = {self.input_name: input_tensor}
        output_tensors = {self.output_name: self.output_tensor}
        self.net.process(self.graph_name, input_tensors, self.input_shapes, output_tensors)
        return self.output_tensor.asnumpy()

    def postprocess(self, embedding):
        """L2 normalize the embedding vector"""
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding / (norm + 1e-10)

    def __call__(self, bmimg_list):
        """Run full pipeline: preprocess -> inference -> postprocess"""
        img_num = len(bmimg_list)
        if self.batch_size == 1:
            start_time = time.time()
            preprocessed_bmimg = self.preprocess_bmcv(bmimg_list[0])
            self.preprocess_time += time.time() - start_time

            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)

            start_time = time.time()
            outputs = self.predict(input_tensor)
            self.inference_time += time.time() - start_time

            start_time = time.time()
            embedding = self.postprocess(outputs)
            self.postprocess_time += time.time() - start_time

            return [embedding]
        else:
            # Batch mode
            BMImageArray = getattr(sail, 'BMImageArray{}D'.format(self.batch_size))
            bmimgs = BMImageArray()
            for i in range(img_num):
                start_time = time.time()
                preprocessed_bmimg = self.preprocess_bmcv(bmimg_list[i])
                self.preprocess_time += time.time() - start_time
                bmimgs[i] = preprocessed_bmimg.data()

            input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
            self.bmcv.bm_image_to_tensor(bmimgs, input_tensor)

            start_time = time.time()
            outputs = self.predict(input_tensor)
            self.inference_time += time.time() - start_time

            start_time = time.time()
            embeddings = self.postprocess(outputs[:img_num])
            self.postprocess_time += time.time() - start_time

            return [embeddings[i:i+1] for i in range(img_num)]


def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))

    # create save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # initialize net
    arcface = ArcFace(args)
    batch_size = arcface.batch_size
    handle = sail.Handle(args.dev_id)
    arcface.init()

    decode_time = 0.0
    cn = 0

    # test images
    if os.path.isdir(args.input):
        bmimg_list = []
        filename_list = []
        all_results = []

        for root, dirs, filenames in os.walk(args.input):
            filenames.sort()
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.jpg', '.png', '.jpeg', '.bmp', '.webp']:
                    continue
                img_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, img_file: {}".format(cn, img_file))

                # decode
                start_time = time.time()
                decoder = sail.Decoder(img_file, True, args.dev_id)
                bmimg = sail.BMImage()
                ret = decoder.read(handle, bmimg)
                if ret != 0:
                    logging.warning("Failed to decode: {}".format(img_file))
                    continue
                decode_time += time.time() - start_time

                bmimg_list.append(bmimg)
                filename_list.append(filename)

                if (len(bmimg_list) == batch_size or cn == len(filenames)) and len(bmimg_list):
                    # predict
                    embeddings = arcface(bmimg_list)

                    for i, filename in enumerate(filename_list):
                        embedding = embeddings[i].flatten()
                        all_results.append({
                            'filename': filename,
                            'embedding': embedding,
                        })
                        logging.info("  {}: embedding shape={}, norm={:.4f}".format(
                            filename, embedding.shape, np.linalg.norm(embedding)))
                        logging.info("Embedding[:5]: {}".format(embedding[:10]))
                    bmimg_list.clear()
                    filename_list.clear()

        # Save results
        result_path = os.path.join(output_dir, 'embeddings.npz')
        names = [r['filename'] for r in all_results]
        embs = np.array([r['embedding'] for r in all_results])
        np.savez(result_path, names=names, embeddings=embs)
        logging.info("Results saved to {}".format(result_path))

    else:
        # single image
        decoder = sail.Decoder(args.input, True, args.dev_id)
        bmimg = sail.BMImage()
        ret = decoder.read(handle, bmimg)
        if ret != 0:
            raise Exception("Failed to decode: {}".format(args.input))
        cn = 1

        embedding = arcface([bmimg])[0].flatten()
        logging.info("Embedding shape: {}, norm: {:.4f}".format(embedding.shape, np.linalg.norm(embedding)))
        logging.info("Embedding[:10]: {}".format(embedding[:10]))

        result_path = os.path.join(output_dir, 'embedding.npy')
        np.save(result_path, embedding)
        logging.info("Result saved to {}".format(result_path))

    # print speed
    decode_time = decode_time / max(cn, 1)
    preprocess_time = arcface.preprocess_time / max(cn, 1)
    inference_time = arcface.inference_time / max(cn, 1)
    postprocess_time = arcface.postprocess_time / max(cn, 1)

    logging.info("------------------ Predict Time Info ----------------------")
    logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
    logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
    logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
    logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))


def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input, images directory or single image')
    parser.add_argument('--bmodel', type=str, default='../models/BM1684X/arcface_resnet50_fp32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')
