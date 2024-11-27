# ChatTTS模型导出与编译

## 1. 准备工作

ChatTTS模型导出需要依赖[ChatTTS huggingface仓库](https://huggingface.co/2Noise/ChatTTS)。

**注意：** 

- 编译模型需要在x86主机完成。

## 2. 主要步骤

### 2.1 TPU-MLIR环境搭建

模型编译前需要安装TPU-MLIR，具体可参考[TPU-MLIR环境搭建](../../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。安装好后需在TPU-MLIR环境中进入例程目录。使用TPU-MLIR将onnx模型编译为BModel。

### 2.2 获取onnx

### 2.2.1 下载模型

**注：** 在下载之前，要确认自己有huggingface官网的access token或者SSH key。
```bash
git lfs install
git clone git@hf.co:2Noise/ChatTTS
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

### 2.2.2 导出onnx

- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可

```bash
pip3 install -r requirements.txt
cd tools
python3 exporter.py --source_dir /path/to/ChatTTS --gpt --decoder --vocos
mkdir -p ../models/onnx
mkdir -p ../models/torch
mv tmp/gpt tmp/*.onnx ../models/onnx
mv tmp/*.pt ../models/torch #decoder模型不是onnx，是torchscript。
cd ..
```

### 2.2 bmodel编译

可以使用如下命令生成bmodel，`gen_gpt_bmodel.sh`的传参形式和其他两个脚本不一样，脚本代码不复杂，可以看脚本中的具体实现。常用的参数都在下面的示例中。

```bash
cd scripts
./gen_gpt_bmodel.sh --mode int4 --seq_length 1024 --target bm1688 --name "chattts-llama"
./gen_decoder_bmodel.sh bm1688
./gen_vocos_bmodel.sh bm1688
```

编译成功之后，模型将会存放在当前目录，请将它们移动到`../models/`目录下。