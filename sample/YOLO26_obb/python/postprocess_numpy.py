import numpy as np
import math

class PostProcess:
    def __init__(self, conf_thresh=0.001):
        self.conf_thresh = conf_thresh
        self.count = 0

    def __call__(self, preds_batch, org_size_batch, ratios_batch, txy_batch):
        """Post-processes predictions and returns a list of Results objects."""
        if isinstance(preds_batch, list) and len(preds_batch) == 1:
            # 1 output
            dets = np.concatenate(preds_batch)
            
        outs = dets[(dets[:, :,4] > self.conf_thresh)]
        if len(outs.shape) == 2:
            outs = np.expand_dims(outs, 0)
            
        results = []
        for pred, (org_w, org_h), ratio, (tx1, ty1) in zip(outs, org_size_batch, ratios_batch, txy_batch):
            rboxes = regularize_rboxes(np.concatenate([pred[:, :4], pred[:, -1:]], axis=-1))
            coords = rboxes[:, :4]
            coords[:, 0] -= tx1  # x padding
            coords[:, 1] -= ty1  # y padding
            coords[:, [0, 2]] /= ratio[0]   # x, w
            coords[:, [1, 3]] /= ratio[1]   # y, h
            rboxes[:, :4] = coords.round()

            # xywh, r, conf, cls
            obb = np.concatenate([rboxes, pred[:, 4:6]], axis=-1)
            results.append(np.ascontiguousarray(obb))
        self.count += 1 
        return results



def regularize_rboxes(rboxes):
    """
    规范化旋转边界框，确保宽度 >= 高度
    
    参数:
        rboxes: ndarray, shape (..., 5), 格式 [x, y, w, h, theta]
    
    返回:
        ndarray, shape (..., 5), 规范化后的旋转框
    """
    # 解包5个分量（支持任意批次维度）
    x = rboxes[..., 0]
    y = rboxes[..., 1]
    w = rboxes[..., 2]
    h = rboxes[..., 3]
    t = rboxes[..., 4]
    
    # 创建掩码：w > h 的位置保持原样，否则需要交换
    mask = w > h
    
    # 根据掩码选择新的宽高
    w_ = np.where(mask, w, h)      # w_ = max(w, h)
    h_ = np.where(mask, h, w)      # h_ = min(w, h)
    
    # 调整角度：当 w <= h 时，角度 + π/2
    t = np.where(mask, t, t + math.pi / 2)
    
    # 角度对 π 取模，归一化到 [0, π)
    t = np.mod(t, math.pi)
    
    # 重新组合成规范化的旋转框
    return np.stack([x, y, w_, h_, t], axis=-1)
    
