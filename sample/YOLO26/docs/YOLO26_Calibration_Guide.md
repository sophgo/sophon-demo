# 模型量化
更多模型量化教程请参考《TPU-MLIR开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

## 1. 注意事项
### 1.1 量化数据集
建议从训练集随机抽取100~500张样本作为量化数据集，量化数据集应尽量涵盖测试场景和类别，量化时可尝试不同的iterations进行量化以获得最优的量化精度。

### 1.2 前处理对齐
量化数据集的预处理应该和推理测试的预处理保持一致，否则会导致较大的精度损失。在mlir制作npz/npy数据集时，应当预先完成数据的预处理。

### 1.3 生成qtable
MLIR具体步骤如下：
1. 可以先用mlir2onnx.py这个工具，将model_transform生成的mlir文件转化成onnx，然后通过netron查看onnx网络结构。
   ```bash
   mlir2onnx.py -m yolo26s_1b.mlir -o yolo26s_mlir.onnx
   ```
2. 使用fp_forward.py生成post.qtable，指定层名即可将对应的所有层指定对应的fp_type。
   ```bash
   fp_forward.py --fpfwd_outputs /model.22/cv2/conv/Conv_output_0_Conv,/model.23/one2one_cv3.1/one2one_cv3.1.0/one2one_cv3.1.0.0/conv/Conv_output_0_Conv,/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv_output_0_Conv,/model.23/one2one_cv3.0/one2one_cv3.0.0/one2one_cv3.0.0.0/conv/Conv_output_0_Conv,/model.23/one2one_cv2.0/one2one_cv2.0.0/conv/Conv_output_0_Conv --chip bm1684x yolo26s_1b.mlir -o post.qtable
   ```
   **注意，在部分版本mlir中，--chip参数或许不支持bm1688/cv186x，您可以使用bm1684x代替，生成的qtable都是通用的，您也可以自由地更改qtable中每一层对应的的fp_type。**

3. 使用run_calibration.py，在生成cali_table的同时自动生成shape_pattern_qtable。
   ```bash
   run_calibration.py yolo26s_1b.mlir \
         --dataset ../datasets/coco128/ \
         --input_num 128 \
         --part_asymmetric \
         --cali_method percentile9999 \
         --fp_type F32 \
         -o yolo26s_cali_table
   ```
   **注意，可以使用shape_pattern_qtable作为编译F16 BModel的qtable。**

4. 将post.qtable与shape_pattern_qtable的敏感层合并到同一个文件，作为量化使用的qtable。
   **注意，post.qtable的敏感层应放在shape_pattern_qtable敏感层的前面，其中重复的敏感层会以shape_pattern_qtable设置的fp_type为准。**

5. 生成的qtable传给model_deploy.py，配合加入test_input和test_reference来验证混精度策略是否有效。