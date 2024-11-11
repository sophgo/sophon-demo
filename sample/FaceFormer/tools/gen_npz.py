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

# 设置目标文件夹路径
test_path = '../models/testInput'

# 检查文件夹是否存在
if not os.path.exists(test_path):
    # 文件夹不存在，创建文件夹
    os.makedirs(test_path)
    print("文件夹已创建：", test_path)
else:
    print("文件夹已存在：", test_path)

random_tensor = np.random.rand(1, 262144).astype(np.float32)
random_tensor2 = np.random.rand(1, 490, 512).astype(np.float32)
one_hot_random_tensor = np.random.rand(1, 8).astype(np.float32)
vertice_input = np.random.rand(1, 490, 64).astype(np.float32)
hidden_states = np.random.rand(1, 490, 64).astype(np.float32)
memory_mask = np.random.rand(490, 490) > 0.5
embedding = np.random.rand(1, 490, 64).astype(np.float32)

np.savez(os.path.join(test_path, 'input_encoder_1.npz'), **{'audio_feature':random_tensor})
np.savez(os.path.join(test_path, 'input_encoder_2.npz'), **{'hidden_states': random_tensor2,'one_hot':one_hot_random_tensor})
np.savez(os.path.join(test_path, 'input_ppe.npz'), **{'embedding': embedding})
np.savez(os.path.join(test_path, 'input_decoder.npz'), **{'vertice_input': vertice_input,'hidden_states':hidden_states,'memory_mask':memory_mask})
print("测试数据已生成：", test_path)