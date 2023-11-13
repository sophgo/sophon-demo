import os
import os.path as osp
from pathlib import Path
import tempfile
import numpy as np
import cv2
import logging
from terminaltables import AsciiTable
import warnings

RESEZE_ORIGIN=0

data = dict(
        data_root='datasets/cityscapes_small/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val')

def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file."""
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name

def scandir(dir_path, suffix=None, recursive=False):
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index=255,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics"""

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics
def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union."""

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float64)
    total_area_union = np.zeros((num_classes, ), dtype=np.float64)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float64)
    total_area_label = np.zeros((num_classes, ), dtype=np.float64)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
        print("process："+str(i)+"/"+str(num_imgs))
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union."""
    
    

    if isinstance(pred_label, str):
        pred_label = cv2.imread(pred_label,cv2.IMREAD_GRAYSCALE)
        # pred_label=np.load(pred_label)

    if isinstance(label, str):
        label = imread_to_array(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label

def imread_to_array(img_or_path):
    # Rest of the code...
        img = cv2.imread(img_or_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img_gray = cv2.resize(img_gray, (1024, 512))
        array2 = np.array(resized_img_gray)
        if array2.ndim >= 3 and array2.shape[2] >= 3:  # color image
            array2[:, :, :3] = array2[:, :, (2, 1, 0)]  # RGB to BGR

        return array2

CLASSES=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                'bicycle')
PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                [0, 80, 100], [0, 0, 230], [119, 11, 32]]


def cityscapes_classes():
    """Cityscapes class names for external use."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]

def cityscapes_palette():
    """Cityscapes palette for external use."""
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]



dataset_aliases = {
    'cityscapes': ['cityscapes'],
    'ade': ['ade', 'ade20k'],
    'voc': ['voc', 'pascal_voc', 'voc12', 'voc12aug']
}


dataset='cityscapes'

def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if isinstance(dataset,str):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels



def get_palette(dataset):
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name
    if isinstance(dataset, str):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got{type(dataset)}')
    return labels

def palette_img(img, result, palette=None):
    
    seg = result[0]
    if palette is None:
        if PALETTE is None:
            palette = np.random.randint(
                0, 255, size=(len(CLASSES), 3))
        else:
            palette = PALETTE
    palette = np.array(palette)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    # from IPython import embed; embed(header='debug vis')
    img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    return img

def palette_bmcv(result, palette=None):
    
    seg = result[0]
    if palette is None:
        if PALETTE is None:
            palette = np.random.randint(
                0, 255, size=(len(CLASSES), 3))
        else:
            palette = PALETTE
    palette = np.array(palette)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]
    return color_seg

def save_and_show_palette_img(palette_img,fig_size=(15, 10),
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    if show:
        _imshow(palette_img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file,palette_img)

    if not (show or out_file):
        warnings.warn('show==False and out_file is not specified, only '
                        'result image will be returned')

# 暂不支持显示
def _imshow(img, win_name='', wait_time=0):
    """Show an image."""
    cv2.imshow(win_name, img)
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)

    
def get_gt_seg_maps(img_infos,ann_dir):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in img_infos:
            seg_map = osp.join(ann_dir, img_info['ann']['seg_map'])
            gt_seg_map = imread_to_array(seg_map)
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps
    
def evaluate(results,
                 metric='mIoU',
                 logger=None,
                 img_infos=None,
                 ann_dir=None):
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
       
        gt_seg_maps = get_gt_seg_maps(img_infos,ann_dir)
        num_classes = len(CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            255,
            metric)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        
        # 用于auto_test
        mAcc=summary_table_data[1][3];
        logging.info("mAcc = {}".format(mAcc))

        # 打印表头       
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)
        


logger_initialized = {}
def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')

