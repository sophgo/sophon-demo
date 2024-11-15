# MP_SENet模型导出与编译

## 主要步骤

模型编译前需要先导出onnx，然后在docker环境中安装TPU-MLIR，安装好后需在docker环境中进入例程目录，使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

### 1. 获取onnx
- 获取MP_SENet的onnx

您需要获取MP_SENet_model，然后将之转换为onnx。
```bash
# 通过官方git链接下载 g_best_dns 或 g_best_vb，代表在两个不同数据上训练的最佳结果
https://github.com/yxlu-0102/MP-SENet/tree/main/best_ckpt
```
如果无法下载，也可以下载我们之前下好的，推荐放置到‘models/torch/’文件夹下
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/models/torch/g_best_dns
python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/models/torch/g_best_vb
```
执行转 ONNX 脚本
```bash
# 对齐环境，如果过程中提示缺少某些组件，直接**pip3 install**组件即可
pip3 install -r tools/requirements_model.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 转ONNX，各项参数请按照您实际情况填写
cd tools
python3 model_onnx.py --checkpoint_file ../models/torch/g_best_dns --output_dir ../models/onnx --config_file ./configs/config.json
```
我们也提供已经转好的onnx文件进行下载，推荐放置到‘models/onnx/’文件夹下
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/models/onnx/mpsenet_dns.onnx
python3 -m dfss --url=open@sophgo.com:sophon-demo/MP_SENet/models/onnx/mpsenet_vb.onnx
```

### 2. TPU-MLIR环境搭建

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。

### 3. BModel编译

目前支持bm1684x编译，使用如下命令生成bmodel。使用时请注意代码中的文件路径是否与当前一致，请按照您实际情况填写。

```bash
cd scripts
./gen_fp32bmodel_mlir.sh bm1684x 
./gen_bf16bmodel_mlir.sh bm1684x 
```

编译成功之后，mp_senet模型的bmodel将会存放在`models/BM1684X/`目录下。

