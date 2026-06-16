```
模型名称:          【填写】
算法类别:          【分类/目标检测/语义分割/姿态估计/人脸识别/语音识别/OCR/立体匹配/超分辨率/多目标跟踪/LLM/图像生成/其他】
原始框架:          【PyTorch/TensorFlow/PaddlePaddle/其他】
模型文件位置:       【本地路径或下载链接】
输入尺寸:          【如 112x112 图像 / 16kHz 音频】
输入通道数:         【如 3 (RGB) / 80 (FBANK)】
输出规格:          【如 512维特征向量 / [1,68,18] CTC输出 / [N,8404] token logits】
预处理方式:         【如图像: Resize→Normalize: mean/std; 音频: FBANK特征提取→CMVN归一化】
模型架构:          【单模型 / 编码器-解码器 / 级联多模型 / Transformer Decoder-Only】
子模型个数:         【1 / 2 / 3 / 更多】
目标芯片:          【BM1684X / BM1688 / CV186X】
目标设备：         【SC7 (用户名@ip:密码) / SE7-32 (用户名@ip:密码) / SE9-16 (用户名@ip:密码) / SE9-8 (用户名@ip:密码)】
需要精度:          【FP32 / FP16 / INT8 / INT8_4b，多选用逗号分隔】
需要 batch:        【1b / 4b / 10b】
需要 Python:       【是/否】
需要 C++:          【是/否】
C++ 前后处理方式:   【bmcv / opencv / 自定义】
bm1684x soc-sdk路径：【绝对路径】
bm1688/cv186x soc-sdk路径:  【绝对路径】 
测试数据集:         【路径或描述】
精度指标:          【如 ACC(%) / COCO mAP / CER+WER(%) / 余弦相似度 / PSNR / D1 / MOTA】
性能指标:          【如 FPS / RTF / tokens/s / 单帧耗时(ms)】
```
