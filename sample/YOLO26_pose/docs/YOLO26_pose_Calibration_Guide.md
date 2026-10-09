# 模型量化
更多模型量化教程请参考《TPU-MLIR开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

## 1. 注意事项
### 1.1 量化数据集
建议从训练集随机抽取100~500张样本作为量化数据集，量化数据集应尽量涵盖测试场景和类别，量化时可尝试不同的iterations进行量化以获得最优的量化精度。

### 1.2 前处理对齐
量化数据集的预处理应该和推理测试的预处理保持一致，否则会导致较大的精度损失。在mlir制作npz/npy数据集时，应当预先完成数据的预处理。

### 1.3 why：YOLO26-pose 为什么需要混合精度量化表
YOLO26-pose 使用端到端输出（`output0 [batch, 300, 57]`），模型头部（本例为 `/model.23`）除了输出层的卷积以外，还包含一条由 `Reshape/Slice/Concat/Mul/Transpose/Split/ReduceMax/TopK/GatherElements/Mod/Expand/Gather/Cast` 等算子组成的候选框筛选链路。这条链路的作用等价于 NMS：
1. `GatherElements` 按候选框置信度取出对应的 17 个关键点；
2. `TopK` 选出置信度最高的 300 个候选框并保持降序；
3. `Mod/Cast/Expand` 等算子完成索引整理，最终拼成 `output0`。

它与常见的检测后处理不同——**没有任何可退化的“数量”误差**：
* FP16：该链路在 F16 下计算时，TopK 的排序结果会发生错乱，AP@[0.5:0.95] 由 0.576 掉到 0.290；
* INT8：该链路的量化误差会把候选框置信度整体压到阈值以下，模型对**所有输入**都输出空结果（1000 张验证图片全部检测不到人体）。

因此本例程为 FP16 与 INT8 各提供了一份混合精度量化表（`models/onnx/yolo26s-pose_qtable_f16`、`models/onnx/yolo26s-pose_qtable_int8`，由 `download.sh` 下载），把上述链路（以及敏感的输出层卷积、其下游层）固定在 F32 计算，其余层仍按 F16/INT8 计算。两张表内容一致，与目标平台无关（BM1684X/BM1688/CV186X 通用）。

### 1.4 how：生成 qtable
MLIR具体步骤如下：
1. 可以先用mlir2onnx.py这个工具，将model_transform生成的mlir文件转化成onnx，然后通过netron查看onnx网络结构。
   ```bash
   mlir2onnx.py -m yolo26s-pose_1b.mlir -o yolo26s-pose_mlir.onnx
   ```
2. 使用fp_forward.py生成post.qtable，指定层名即可将对应的所有层指定对应的fp_type。YOLO26-pose 的敏感层集中在头部输出卷积及其下游：
   ```bash
   fp_forward.py --fpfwd_outputs /model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_Conv,/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0_Conv,/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0_Conv,/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0_Conv,/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0_Conv,/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0_Conv,/model.23/one2one_cv4_kpts.0/Conv_output_0_Conv,/model.23/one2one_cv4_kpts.1/Conv_output_0_Conv,/model.23/one2one_cv4_kpts.2/Conv_output_0_Conv --chip bm1684x yolo26s-pose_1b.mlir -o post.qtable
   ```
   **注意，在部分版本mlir中，--chip参数或许不支持bm1688/cv186x，您可以使用bm1684x代替，生成的qtable都是通用的，您也可以自由地更改qtable中每一层对应的的fp_type。**

3. 使用run_calibration.py，在生成cali_table的同时自动生成shape_pattern_qtable。
   ```bash
   run_calibration.py yolo26s-pose_1b.mlir \
         --dataset ../datasets/coco128/ \
         --input_num 128 \
         --part_asymmetric \
         --cali_method percentile9999 \
         --fp_type F32 \
         -o yolo26s-pose_cali_table
   ```
   **注意，`shape_pattern_qtable` 已经包含第 2 步指定的全部 9 个头部输出卷积，可以直接作为编译 F16 BModel 的 qtable。**它按算子 pattern 把端到端的 TopK/GatherElements 链路连同这些输出卷积一并标为 F32（共 62 层）；但它只反映 1b mlir 的结构，编译 INT8（尤其是 `batch_size=4`）BModel 时还需要合并第 4 步中只在 4b mlir 里出现的算子。

4. 将post.qtable与shape_pattern_qtable的敏感层合并到同一个文件，作为量化使用的qtable。
   **注意，post.qtable的敏感层应放在shape_pattern_qtable敏感层的前面，其中重复的敏感层会以shape_pattern_qtable设置的fp_type为准。**

   本例程提供的 `models/onnx/yolo26s-pose_qtable_int8`（FP16 使用内容相同的 `models/onnx/yolo26s-pose_qtable_f16`）即为合并结果，共 64 层：

   ```
   # op_name   quantize_mode
   /model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_Conv F32
   ...
   /model.23/TopK_output_0_TopK F32
   /model.23/GatherElements_output_0_GatherElements F32
   /model.23/Mod_output_0_Mod F32
   ...
   /model.23/Cast_9_output_0_Cast F32
   output0_Concat F32
   ```

   其中 62 层由 `shape_pattern_qtable` 自动识别得到；另外 2 层 `/model.23/Mul_8_output_0`、`/model.23/Add_9_output_0_Add` 只在 `batch_size=4` 的 mlir 中出现（1b 的 mlir 里没有这两个算子），手工补充以保证 4b 模型也能正常编译。注意 qtable 中的层名必须与 mlir 里的 `loc` 名**完全一致**（`model_deploy.py` 对匹配不上的层名既不报错也不告警，会直接忽略），`Mul_8` 在 4b 里已被折叠为常量（`top.Weight`），其 `loc` 名不带算子后缀。

5. 生成的qtable传给model_deploy.py，配合加入test_input和test_reference来验证混精度策略是否有效。

   ```bash
   model_deploy.py \
       --mlir yolo26s-pose_1b.mlir \
       --quantize INT8 \
       --chip cv186x \
       --calibration_table yolo26s-pose_cali_table \
       --quantize_table yolo26s-pose_qtable_int8 \
       --model yolo26s-pose_int8_1b.bmodel
   ```

   验证混精度策略是否有效，可以在编译后重新跑一遍 6.1 节的精度测试（AP 不应大幅低于 FP32）；如果不想跑完整评测，用视频测试看每帧的检测数量也可以快速判断：
   ```bash
   python3 python/yolo26_pose_bmcv.py --input datasets/dance_1080P.mp4 \
       --bmodel models/CV186X/yolo26s-pose_int8_1b.bmodel --dev_id 0
   ```
   正常时日志里会持续打印 `xxx, det nums: n`（n > 0）；如果 qtable 没有正确传给 `model_deploy.py`，所有帧都会是 `det nums: 0`（视频测试只输出 `results/images/` 下的可视化结果，不写结果 json）。
