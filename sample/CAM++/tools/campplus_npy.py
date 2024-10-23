#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import numpy as np
import os

def generate_npy(path):
    sizes = [[1, 600, 80]]
    keys = ['feature']

    for i in range(10):
        arrays_dict = {}
        for key, size in zip(keys, sizes):
            arrays_dict[key] = np.random.rand(*size).astype(np.float32)

        np.savez(path+'/input_'+str(i)+'.npz', **arrays_dict)

if __name__ == '__main__':
    if not os.path.exists("../datasets"):
        os.mkdir("../datasets")
    if not os.path.exists("../datasets/cali_set_npy"):
        os.mkdir("../datasets/cali_set_npy")
    generate_npy("../datasets/cali_set_npy")
