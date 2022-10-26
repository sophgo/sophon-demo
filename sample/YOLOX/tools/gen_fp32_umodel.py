#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import ufw.tools as tools
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for resize")
    parser.add_argument('--trace_model', type=str, default="/workspace/test/YOLOX/models/yolox_s.trace.pt")
    parser.add_argument('--data_path',type=str, default="/workspace/test/YOLOX/datasets/ost_data_enhance/data.mdb")
    parser.add_argument('--dst_width',type=int, default=640)
    parser.add_argument('--dst_height',type=int, default=640)

    opt = parser.parse_args()
    
    (filepath, filename) = os.path.split(opt.trace_model)
    save_path = filepath+"/../middlefiles"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ptyolox = [
    '-m', '{}'.format(opt.trace_model),
    '-s', '(1,3,{},{})'.format(opt.dst_width,opt.dst_height),
    '-d', '{}'.format(save_path),
    '-D', '{}'.format(opt.data_path),
    '--cmp'
]

    tools.pt_to_umodel(ptyolox)