# @Time    : 2023/8/28 22:28
# @Author  : zhangchenming

import torch
import random
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_params_from_file(model, filename, device, dist_mode, logger, strict=True):
    checkpoint = torch.load(filename, map_location=device)
    pretrained_state_dict = checkpoint['model_state']
    tmp_model = model.module if dist_mode else model
    state_dict = tmp_model.state_dict()

    unused_state_dict = {}
    update_state_dict = {}
    unupdate_state_dict = {}
    for key, val in pretrained_state_dict.items():
        if key in state_dict and state_dict[key].shape == val.shape:
            update_state_dict[key] = val
        else:
            unused_state_dict[key] = val
    for key in state_dict:
        if key not in update_state_dict:
            unupdate_state_dict[key] = state_dict[key]

    if strict:
        tmp_model.load_state_dict(update_state_dict)
    else:
        state_dict.update(update_state_dict)
        tmp_model.load_state_dict(state_dict)

    message = 'Unused weight: '
    for key, val in unused_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)

    message = 'Not updated weight: '
    for key, val in unupdate_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)