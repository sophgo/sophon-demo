"""加载 GLC_Image 权重并导出熵编码所需资产。

GLC 源码里 `from torchvision import models` 是无用 import，而当前环境没有可用的
匹配 torchvision 轮子，所以需要把 stubs 目录排在 sys.path 前面。
"""

import os
import sys


def prepare_paths(glc_src_parent, stub_dir=None):
    """把 GLC 源码父目录与 torchvision stub 依次插入 sys.path。"""
    if stub_dir is not None and stub_dir not in sys.path:
        sys.path.insert(0, stub_dir)
    if glc_src_parent not in sys.path:
        sys.path.insert(0, glc_src_parent)


def load_glc(ckpt_path, glc_src_parent, stub_dir=None, inplace=False):
    """加载 GLC_Image 并返回 eval 模式的模型。

    权重按 strict=True 加载：GLC_image.pth.tar 是官方原版权重，
    如果键不匹配说明源码版本与权重不是同一套，必须报错而不是静默忽略。
    """
    prepare_paths(glc_src_parent, stub_dir)
    import torch

    from src.models.image_model import GLC_Image
    from src.utils.test_utils import get_state_dict

    net = GLC_Image(inplace=inplace)
    state = get_state_dict(ckpt_path)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net
