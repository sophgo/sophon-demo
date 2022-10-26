#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from loguru import logger

import numpy as np
import sophon.sail as sail
import inspect

"""
    analysis input var list and get its name. See https://github.com/jinfagang/alfred/blob/main/alfred/dl/torch/common.py
    :param var: variables like "var1, var2, var3" to get name from.
"""
def decorator(f):
    def wrapper(*args, **kwargs):
        bound_args = inspect.signature(f).bind(*args, **kwargs)
        bound_args.apply_defaults()

        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args_ori_names = string[string.find('(') + 1:-1].split(',')

        names = []
        for i in args_ori_names:
            if i.find('=') != -1:
                names.append(i.split('=')[1].strip())
            else:
                names.append(i)
        args_dict = dict(zip(names, args))
        for k, v in args_dict.items():
            k = k.strip()
            if isinstance(v, sail.BMImage):
                logger.debug('[ <sail.BMImage> {}]: {}, {}, {}', k, v.format(), v.dtype(), (v.width(), v.height()))
            else:
                logger.debug('[ <numpy.ndarray> {}]: {}', k, v.shape)
        return f(*args, **kwargs)
    return wrapper

@decorator
def print_infos(*vs):
    pass

# see https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/40536047#40536047
def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

def print_info(image):
    if isinstance(image, sail.BMImage):
        logger.debug('[ BMIMage - {}]: {}, {}, {}', retrieve_name(image), image.format(), image.dtype(), (image.width(), image.height()))
    elif isinstance(image, np.ndarray):
        logger.debug('[ numpy - {}]: {}, {}, {}', retrieve_name(image), image.dtype, image.shape)


if __name__ == '__main__':
    bm_handle = sail.Handle(0)
    bm_image1 = sail.BMImage(bm_handle, 1080, 1920, sail.Format.FORMAT_RGB_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
    bm_image2 = sail.BMImage(bm_handle, 1080, 1920, sail.Format.FORMAT_RGB_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
    bm_image3 = sail.BMImage(bm_handle, 1080, 1920, sail.Format.FORMAT_RGB_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
    bm_image4 = sail.BMImage(bm_handle, 1080, 1920, sail.Format.FORMAT_RGB_PLANAR, sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE)
    logger.debug("original image: format = {}, (width,height) = {}".format( \
            bm_image1.format(), (bm_image1.width(), bm_image1.height())))
    print_infos(bm_image1, bm_image2, bm_image3, bm_image4)
    print_info(bm_image1)