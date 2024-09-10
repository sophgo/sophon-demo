# VITS_CHINESE模型导出与编译

## 主要步骤

模型编译前需要先导出onnx，然后在docker环境中安装TPU-MLIR，安装好后需在docker环境中进入例程目录，使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)相应版本的SDK中获取)。

### 1. 获取onnx
- 获取BERT的onnx
```bash
wget https://github.com/wangyifan2018/VITS-TPU/releases/download/v3.0/bert.onnx
```
如果无法下载，也可以下载我们之前下好的
```bash
 python3 -m dfss --url=open@sophgo.com:sophon-demo/VITS_CHINESE/models/bert.onnx
```
- 获取VITS的onnx

您需要获取vits_bert_model.pth，然后将之转换为onnx。
```bash
# 下载 vits_bert_model.pth
wget https://github.com/PlayVoice/vits_chinese/releases/download/v1.0/vits_bert_model.pth
```
如果无法下载，也可以下载我们之前下好的
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/VITS_CHINESE/models/vits_bert_model.pth
```
执行转 ONNX 脚本
```bash
# 转ONNX需要用到第三方text库，因此需要指定库的路径
export PYTHONPATH=$PWD/python:$PYTHONPATH
# 对齐环境，如果过程中提示缺少某些组件，直接**pip3 install**组件即可
pip3 install -r tools/requirements_model.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 转ONNX，--model 的目录请按照您实际情况填写
python3 tools/model_onnx.py --config tools/configs/bert_vits.json --model vits_bert_model.pth
```

### 2. TPU-MLIR环境搭建

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。

### 3. BModel编译

目前TPU-MLIR支持bm1684x和bm1688、cv186x编译BERT、VITS，使用如下命令生成bmodel。

```bash
./scripts/gen_bmodel.sh bm1684x #bm1688#cv186x
```

编译成功之后，vits模型和bert模型的bmodel将会存放在`models/BM1684X/`目录下。


