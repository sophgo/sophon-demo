import os
import cv2
import clip
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def topk(x, k):
    indices = np.argpartition(x, -k)[-k:]
    indices = indices[np.argsort(-x[indices])]
    return x[indices], indices

def main(args):
    # Load bmodel
    text = args.text
    model, preprocess = clip.load(args.image_model, args.text_model, args.dev_id)
    
    # Tokenize 
    text_inputs = clip.tokenize(text)

    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    else:
        image_paths = [os.path.join(args.image_path, fname) for fname in os.listdir(args.image_path)]
    
    for filename in image_paths:
        image = cv2.imread(filename)
        image_input = np.expand_dims(preprocess(image), axis=0)


        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= np.linalg.norm(image_features,axis=-1, keepdims=True)
        text_features /= np.linalg.norm(text_features,axis=-1, keepdims=True)

        similarity = softmax((100.0 * np.dot(image_features , text_features.T)),axis=-1) #计算相似度，并转换为概率分布  
        values, indices = topk(similarity[0],min(len(text), 5))

        for i in range(len(text)):
            logging.info(f"Text: {text[indices[i]]}, Similarity: {values[i].item()}")


    logging.info(("-------------------Image num {}, Preprocess average time ------------------------").format(image_input.shape[0]))
    logging.info("preprocess(ms): {:.2f}".format(model.preprocess_time / image_input.shape[0] * 1000))

    logging.info(("------------------ Image num {}, Image Encoding average time ----------------------").format(image_input.shape[0]))
    logging.info("image_encode(ms): {:.2f}".format(model.encode_image_time / image_input.shape[0] * 1000))

    logging.info(("------------------ Image num {}, Text Encoding average time ----------------------").format(image_input.shape[0]))
    logging.info("text_encode(ms): {:.2f}".format(model.encode_text_time / image_input.shape[0] * 1000))



def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--image_path', type=str, default='./datasets/CLIP.png', help='path of input')
    parser.add_argument('--text', nargs='+', default=['a diagram', 'a dog', 'a cat'], help='text of input')
    parser.add_argument('--image_model', type=str, default='./models/BM1684X/clip_image_vitb32_bm1684x_f16_1b.bmodel', help='path of image bmodel')
    parser.add_argument('--text_model', type=str, default='./models/BM1684X/clip_text_vitb32_bm1684x_f16_1b.bmodel', help='path of text bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')